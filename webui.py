import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

# Backend functions from original file are kept
def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue
                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return

def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options
    if mode == modules.flags.inpaint_option_detail:
        return [gr.update(visible=True), gr.update(visible=False, value=[]), False, 'None', 0.5, 0.0]
    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version
    if mode == modules.flags.inpaint_option_modify:
        return [gr.update(visible=True), gr.update(visible=False, value=[]), True, inpaint_engine_version, 1.0, 0.0]
    return [gr.update(visible=False, value=''), gr.update(visible=True), False, inpaint_engine_version, 1.0, 0.618]


reload_javascript()

# Changed title here
title = f'photoshop AI'

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    with gr.Row():
        with gr.Column(scale=2):
            # Main image display area
            with gr.Row():
                progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768, elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain', height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False, elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768, elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'], elem_id='final_gallery')
            
            # Prompt and Generate button area
            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here for inpainting.", elem_id='positive_prompt', autofocus=True, lines=3)
                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)
                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'stop'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask
                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')

            # REMOVED: Checkboxes for Input Image, Enhance, Advanced
            
            # Inpainting panel is now always visible
            with gr.Row(visible=True) as image_input_panel:
                with gr.Row():
                    with gr.Column():
                        inpaint_input_image = grh.Image(label='Image for Inpainting', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=True)
                        inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                        inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                        outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                        gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                        
                        # Inpaint-specific developer tools are kept for full control
                        with gr.Accordion("Inpaint Settings", open=False):
                            inpaint_engine = gr.Dropdown(label='Inpaint Engine', value=modules.config.default_inpaint_engine_version, choices=flags.inpaint_engine_versions, info='Version of Fooocus inpaint model.')
                            inpaint_strength = gr.Slider(label='Inpaint Denoising Strength', minimum=0.0, maximum=1.0, step=0.001, value=1.0)
                            inpaint_respective_field = gr.Slider(label='Inpaint Respective Field', minimum=0.0, maximum=1.0, step=0.001, value=0.618)
                            inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate', minimum=-64, maximum=64, step=1, value=0)
                            invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)

                    # REMOVED: All other tabs (Upscale, Image Prompt, Describe, Enhance, Metadata)

        # REMOVED: The entire right-side column with 'Settings', 'Styles', 'Models', and 'Advanced' tabs has been removed.

    # To avoid breaking the backend, we create hidden state objects for the removed UI components
    # These will pass default values to the generation task
    state_is_generating = gr.State(False)
    negative_prompt = gr.State(modules.config.default_prompt_negative)
    style_selections = gr.State(modules.config.default_styles)
    performance_selection = gr.State(modules.config.default_performance)
    aspect_ratios_selection = gr.State(modules.config.default_aspect_ratio)
    image_number = gr.State(1)
    output_format = gr.State(modules.config.default_output_format)
    image_seed = gr.State(random.randint(constants.MIN_SEED, constants.MAX_SEED))
    read_wildcards_in_order = gr.State(False)
    sharpness = gr.State(modules.config.default_sample_sharpness)
    guidance_scale = gr.State(modules.config.default_cfg_scale)
    base_model = gr.State(modules.config.default_base_model_name)
    refiner_model = gr.State(modules.config.default_refiner_model_name)
    refiner_switch = gr.State(modules.config.default_refiner_switch)
    lora_ctrls = [gr.State(val) for item in modules.config.default_loras for val in item]
    
    # Dummy states for other input modes
    uov_method = gr.State(flags.uov_list[0])
    uov_input_image = gr.State(None)
    ip_ctrls = [gr.State(None)] * (modules.config.default_controlnet_image_count * 4) # Simplified dummy state

    # The inpaint controls that are still visible in the UI
    inpaint_ctrls = [inpaint_engine, invert_mask_checkbox, inpaint_erode_or_dilate, inpaint_strength, inpaint_respective_field]
    inpaint_disable_initial_latent = gr.State(False) # Keep this as a state
    
    # Event handler for the inpaint mode dropdown
    inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
        inpaint_additional_prompt, outpaint_selections,
        inpaint_disable_initial_latent, inpaint_engine,
        inpaint_strength, inpaint_respective_field
    ], show_progress=False, queue=False)

    # Building the list of controls to pass to the backend.
    # It must match the original order and number of arguments.
    # Many arguments are now fed by the hidden gr.State components.
    ctrls = [currentTask, gr.State(False)] # generate_image_grid
    ctrls += [
        prompt, negative_prompt, style_selections,
        performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
        read_wildcards_in_order, sharpness, guidance_scale
    ]
    ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
    ctrls += [gr.State(True), gr.State('inpaint')] # input_image_checkbox, current_tab
    ctrls += [uov_method, uov_input_image]
    ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, gr.State(None)] # inpaint_mask_image is not used directly here
    ctrls += [gr.State(False)] * 4 # disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw
    # Fill in the rest with dummy states for simplicity
    ctrls += [gr.State(val) for val in [1.5, 0.8, 0.3, 4.0, 2]] # adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip
    ctrls += [gr.State(val) for val in [modules.config.default_sampler, modules.config.default_scheduler, modules.flags.default_vae]] # sampler, scheduler, vae
    ctrls += [gr.State(-1)] * 8 # overwrite_step through mixing_image_prompt_and_inpaint
    ctrls += [gr.State(False), gr.State(False), gr.State(64), gr.State(128)] # debugging_cn_preprocessor to canny_high_threshold
    ctrls += [gr.State('joint'), gr.State(0.25)] # refiner_swap_method, controlnet_softness
    ctrls += [gr.State(False), gr.State(1.01), gr.State(1.02), gr.State(0.99), gr.State(0.95)] # free_u
    # The visible inpaint controls are a subset, we need to add the other expected ones as states
    ctrls += [gr.State(False), inpaint_disable_initial_latent] + inpaint_ctrls[:1] + inpaint_ctrls[3:] + [gr.State(True)] + inpaint_ctrls[1:3]
    # Remaining controls for enhance, etc., filled with dummy states
    if not args_manager.args.disable_image_log:
        ctrls += [gr.State(modules.config.default_save_only_final_enhanced_image)]
    if not args_manager.args.disable_metadata:
        ctrls += [gr.State(modules.config.default_save_metadata_to_images), gr.State(modules.config.default_metadata_scheme)]
    ctrls += ip_ctrls
    ctrls += [gr.State(False)] * 3 # debugging_dino, dino_erode_or_dilate, debugging_enhance_masks
    ctrls += [gr.State(None), gr.State(False)] + [gr.State(val) for val in flags.uov_list] # enhance controls
    ctrls += [gr.State(False)] * (modules.config.default_enhance_tabs * 16) # More enhance controls

    # Generate button click event
    generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), True),
                          outputs=[stop_button, generate_button, state_is_generating]) \
        .then(fn=lambda: random.randint(constants.MIN_SEED, constants.MAX_SEED), outputs=image_seed) \
        .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
        .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
        .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), False),
              outputs=[generate_button, stop_button, state_is_generating])

# Final launch code
shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)

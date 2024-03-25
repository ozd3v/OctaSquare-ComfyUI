import os
import torch
from datetime import datetime
import json
import folder_paths
import comfy.sd
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from nodes import MAX_RESOLUTION

def parse_checkpoint_name(ckpt_name):
    return os.path.basename(ckpt_name)


def parse_checkpoint_name_without_extension(ckpt_name):
    return os.path.splitext(parse_checkpoint_name(ckpt_name))[0]


def get_timestamp(time_format): 
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

def make_pathname(filename, seed, modelname, time_format):
#make_pathname(filename, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", parse_checkpoint_name(modelname))
    filename = filename.replace("%seed", str(seed))
    #filename = filename.replace("%counter", str(counter))
    #filename = filename.replace("%sampler_name", sampler_name)
    #filename = filename.replace("%steps", str(steps))
    #filename = filename.replace("%cfg", str(cfg))
    #filename = filename.replace("%scheduler", scheduler)
    #filename = filename.replace("%basemodelname", parse_checkpoint_name_without_extension(modelname))
    #filename = filename.replace("%denoise", str(denoise))
    return filename

def make_filename(filename, seed, modelname, time_format):
    filename = make_pathname(filename, seed, modelname, time_format)
    return get_timestamp(time_format) if filename == "" else filename


def save_json(image_info, filename):
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            print('No image info found, skipping saving of JSON')
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            print(f'Saved workflow to {filename}.json')
    except Exception as e:
        print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')



class StringLiteral:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_string"
    CATEGORY = "OctaSquare/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"default": "", "multiline": True})}}

    def get_string(self, string):
        return (string,)


class CheckpointLoaderWithName:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "modelname",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "OctaSquare/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
               }
        }

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # add checkpoint name to the output tuple (without the ClipVisionModel)
        out = (*out[:3], ckpt_name)
        return out

class SamplerSelector:
    CATEGORY = 'OctaSquare/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (comfy.samplers.KSampler.SAMPLERS,)}}

    def get_names(self, sampler_name):
        return (sampler_name,)
class VAELoaderwithName:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)  
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(), )}}
    RETURN_TYPES = ("VAE","STRING",)
    RETURN_NAMES = ("VAE", "vae_name",)
    FUNCTION = "load_vae"

    CATEGORY = "OctaSquare/utils"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        out = (vae, vae_name)  
        return out

class SaveImageOctaSquare:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "path": ("STRING", {"default": "%date", "multiline": False}),
                     "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                     "info": ("INFOIMG", ),  
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "OctaSquare"

    def save_images(self, images, 
                    path,
                    time_format,
                    info,
                    filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        
        print(f"info to {info}")
        model_name = info['model_name']
        seed = info['seed']
        path = make_pathname(path, seed, model_name,time_format) 
        timeprefix = get_timestamp(time_format)
        
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            filenameout = f"{timeprefix}_{filename_with_batch_num}_{counter:05}_{seed}"
            file = f"{filenameout}.png"
            img.save(os.path.join(output_path, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            self.save_info(info, filenameout, output_path)
            counter += 1

        return { "ui": { "images": results } }  
    
    def save_info(self, info, filename, output_path):
        info["ComfyUI"] = True
        txt_path = os.path.join(output_path, f"{filename}.txt")
        info_copy = info.copy()
        positive_prompt = info_copy.pop('positive_prompt', None) 
        negative_prompt = "Negative prompt: " + info_copy.pop('negative_prompt', None)
        steps = info_copy.pop('Steps', None)
        with open(txt_path, 'w') as f:
            f.write(positive_prompt + '\n')
            f.write(negative_prompt + '\n')
            # Concatena las entradas restantes en 'info' en el formato 'key: value, key: value, ...'
            info_str = ', '.join(f'{key}: {value}' for key, value in info_copy.items())
            info_str = f'Steps: {steps}, ' + info_str
            f.write(info_str)
        return filename

class SettingsOctaSquare:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    RETURN_TYPES = ("INT","LATENT",     "INFOIMG","STRING","STRING",   "INT","FLOAT",comfy.samplers.KSampler.SAMPLERS,comfy.samplers.KSampler.SCHEDULERS,"FLOAT")
    RETURN_NAMES = ("SEED","latent_image","info","positive","negative","steps","cfg","sampler_name","scheduler","denoise")

    FUNCTION = "get_seed"
    CATEGORY = "OctaSquare"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": 
                {
                "model_name": ("STRING", {"default": ""}),
                "vae_name": ("STRING", {"default": ""}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                }
            }

    def get_seed(self, model_name, vae_name, positive_prompt, negative_prompt,  seed, width, height, 
                 steps,
                 cfg,
                 sampler_name,
                 scheduler,
                 denoise,
                 batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        info = {"width":width,"height":height,"batch_size":batch_size,
                 "model_name":model_name,"vae_name":vae_name,
                 "seed":seed,
                 "positive_prompt":positive_prompt,
                 "negative_prompt":negative_prompt,
                 "Steps":steps,
                 "cfg":cfg,
                 "sampler_name":sampler_name,
                 "scheduler":scheduler,
                 "denoise":denoise,
                 }
        return (seed,{"samples":latent},info,positive_prompt,negative_prompt,
                steps,cfg,sampler_name,scheduler,denoise)
    
   

NODE_CLASS_MAPPINGS = {
    "Checkpoint Loader (OctaSquare)": CheckpointLoaderWithName,
    "VAE Loader (OctaSquare)": VAELoaderwithName,
    "Sampler Selector (OctaSquare)": SamplerSelector,
    "Positive Prompt (OctaSquare)": StringLiteral,
    "Negative Prompt (OctaSquare)": StringLiteral,
    "Save Image (OctaSquare)": SaveImageOctaSquare,
    "Settings (OctaSquare)": SettingsOctaSquare,
}

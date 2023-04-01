import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import argparse

MODEL_ID = "timbrooks/instruct-pix2pix"
#PIPE = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
PIPE = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID).to("cpu")

def main(prompt, imagePath):
    image = PIL.Image.open(imagePath)

    images = PIPE(prompt, image = image, num_inference_steps = 20, image_guidance_scale = 1.5, guidance_scale = 7).images

    new_image = PIL.Image.new("RGB", (image.width * 2, image.height))
    new_image.paste(image, (0, 0))
    new_image.paste(images[0], (image.width, 0))

    new_image.save("output.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to image file")
    ap.add_argument("-p", "--prompt", required = True, help = "Prompt for image editing")

    args = vars(ap.parse_args())

    main(args["prompt"], args["image"])

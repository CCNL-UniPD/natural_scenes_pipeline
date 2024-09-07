import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image, SafetySetting, HarmCategory,HarmBlockThreshold
import asyncio
import aiofiles
import os
from tqdm.auto import tqdm
import json
from vertexai.generative_models import GenerativeModel, Image
from asyncio import Semaphore

vertexai.init(project='your-project-id', location="location-close-to-you")
model = GenerativeModel(model_name="gemini-1.0-pro-vision")

image_folder = '.../pascal/VOCdevkit/VOC2012/JPEGImages' # modify here, download the pascal image set first than make sure the path ends like this

# Retrieve absolute paths and filenames without extensions
im_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]
filenames = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.endswith('.jpg')]

# Initialize the multimodal model
multimodal_model = GenerativeModel("gemini-pro-vision")

prompt = (
    "What objects/things are there in the image? "
    "Answer me with only their singular name and if there are more than one object, separate with commas (i.e., ). "
    "Focus on the countable objects, ignore backgrounds and continuous stuff such as sky, lawn, mountain, etc... "
    "If there are many objects of the same kind, name only once. E.g. a scene with 3 apples and 2 banana, you should only answer: apple, banana."
    "Add an Identification (i.e. 77592) code before answering"
)

# Safety config
safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),]

results = {}

# Setup semaphore for concurrency control
max_concurrent_tasks = 20
rate_limit = 0.05
semaphore = Semaphore(max_concurrent_tasks)

def load_image_from_file(file_path):
    return Image.load_from_file(file_path)

async def process_image(im_path, filename):
    async with semaphore:
        counter = 0
        success = False
        backoff = 1
        while counter < 3 and not success:
            try:
                # Load image asynchronously to avoid blocking the event loop
                image = await asyncio.to_thread(load_image_from_file, im_path)
                response = await asyncio.to_thread(multimodal_model.generate_content, [prompt, image], safety_settings=safety_config)
                await asyncio.sleep(rate_limit)
                return filename, response.text
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                counter += 1
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff
                if counter >= 3:
                    return filename, None  

async def main():
    tasks = []
    for im_path, filename in zip(im_paths, filenames):
        tasks.append(process_image(im_path, filename))
    
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        filename, result = await future
        results[filename] = result

    results_file_path = 'save_path'
    async with aiofiles.open(results_file_path, 'w') as json_file:
        await json_file.write(json.dumps(results))
    
    print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    asyncio.run(main())
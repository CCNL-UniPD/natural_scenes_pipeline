"""
Description: This script uses the Gemini vision model from Vertex AI to perform object detection on images from the MSCOCO dataset.
             It prompts the model to identify objects in each image by sending image URLs to the Gemini model. The results, including
             detected objects, are stored in JSON files, with intermediate saves for large datasets. The script includes retry logic 
             for failed requests and reprocessing of erroneous entries after the initial run.
             
Main Features:
1. Asynchronously fetches images from URLs in the MSCOCO dataset and prompts the Gemini model to detect objects.
2. Handles concurrency with asyncio and applies rate-limiting to avoid overloading the API.
3. Supports retry logic for failed requests and reprocesses entries with errors.
4. Saves intermediate and final results in JSON format, allowing safe recovery and progress tracking during large batch processing.
5. Includes functionality to automatically retry and save responses for images that failed in the first attempt.

Author: Kuinan
Date: 2024-09-01
Version: 1.0
"""

import vertexai
from vertexai.preview.generative_models import GenerativeModel
vertexai.init(project='your-project-id', location="location-close-to-you")

from vertexai.generative_models import GenerativeModel
model = GenerativeModel(model_name="gemini-1.0-pro-vision")
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import json
import asyncio
import os
from func import load_image_from_url_V

target_path = 'where you want to save'
autofill_target_path = 'where you want to save after retry'
ann_path = 'path to the MSCOCO annotations. so that we can use url to prompt gemini without downloading all images'
with open(ann_path, 'r') as file:
  lines = json.load(file)

# Optional: Slice the first 100 items for testing
# lines['images'] = lines['images'][:435]

async def process(im_url):
    prompt = ("What objects/things are there in the image?"
                "Answer me with only their singular name and if there are more than one object, separate with commas (i.e., )."
                "If there are many objects of the same kind, name only once. E.g. a scene with 3 apples and 2 banana, you should only answer: apple, banana."
                "Focus on the countable objects, ignore backgrounds and continuous stuff such as sky, lawn, mountain, etc..."
                "Add an Identification (i.e. 77592) code before answering")
    
    im_bytes = await asyncio.to_thread(load_image_from_url_V, im_url)
    response = await asyncio.to_thread(model.generate_content, [prompt, im_bytes])
    await asyncio.sleep(0.5)
    return response.text

async def fetch_and_process(im, results, rate_limit_interval, semaphore, lock):
    im_id = im['id']
    async with semaphore:  # Control concurrency with a semaphore
        counter = 0
        while counter < 3:
            try:
                response = await asyncio.wait_for(process(im['coco_url']), timeout=10)  # Timeout set
                await asyncio.sleep(rate_limit_interval)  # Rate limiting

                # Lock only when accessing shared state
                async with lock:
                    if im_id not in results:
                        results[im_id] = {'url': im['coco_url'], 'success': False}
                    if response:  # Simplified check
                        results[im_id]['response'] = response
                        results[im_id]['success'] = True
                        #print(f"Successfully processed image ID {im_id}.")
                        break
            except asyncio.TimeoutError:
                print(f"Timeout occurred for image ID {im_id}.")
                break
            except Exception as e:
                print(f"Error processing image ID {im_id}: {e}")
                counter += 1

        if counter == 3:
            async with lock:
                if im_id in results:
                    results[im_id]['success'] = False
                    results[im_id]['error'] = "Failed after 3 attempts."
                else:
                    results[im_id] = {'url': im['coco_url'], 'success': False, 'response': None}


async def save_results(results, file_path, lock):
    if not results:  # Early exit if no results to save
        print("No new results to save.")
        return

    # Serialize JSON outside of the locked section to reduce time spent under lock
    print('start dumping results to json...')
    serialized_results = json.dumps(results)
    print('dumping finished.')
    try:
        print('save lock running...')
        async with lock:  # Minimize time in this critical section
            print('save lock finished.')
            try:
                # Only open and read if the file exists and needs to be updated
                if os.path.exists(file_path):
                    print('path correct, start saving...')
                    with open(file_path, 'r+') as file:
                        existing_data = json.load(file)
                        existing_data.update(json.loads(serialized_results))  # Update with new results
                        file.seek(0)
                        json.dump(existing_data, file)
                        file.truncate()
                else:
                    with open(file_path, 'w') as file:
                        file.write(serialized_results)
                print("Results successfully saved.")
            except json.JSONDecodeError:
                # Handle possible JSON decode error by writing a new file
                with open(file_path, 'w') as file:
                    file.write(serialized_results)
                print("JSON decode error; new file created.")
    except Exception as e:
        print(f"Failed to save results: {e}")

async def main(lines, rate_limit_interval, max_concurrent_tasks, save_interval):
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    lock = asyncio.Lock()
    tasks = [fetch_and_process(im, results, rate_limit_interval, semaphore, lock) for im in lines['images']]

    try:
        completed_tasks = 0
        # Use asyncio.as_completed to process tasks as they finish
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task  # Wait for one task to complete
            completed_tasks += 1

            # Save results at the specified interval
            if completed_tasks % save_interval == 0:
                print(f"Intermediate save after {completed_tasks} tasks.")
                await save_results(results, target_path, lock)
                results.clear()  # Clear the results dictionary to avoid duplication

        # After all tasks are done, check if there are any remaining results to save
        if results:
            print("Saving final batch of results...")
            await save_results(results, target_path, lock)
            results.clear()
            print("All tasks completed and final results saved!")
    except Exception as e:
        print(f"Error during task execution: {e}")


print('Start requesting from GEMINI')
asyncio.run(main(lines, rate_limit_interval=0.5, max_concurrent_tasks=5, save_interval=200))

# Reprocess the entries with errors one by one
async def reprocess_errors(results, rate_limit_interval):
    semaphore = asyncio.Semaphore(0.5)  # Only allow one task at a time
    lock = asyncio.Lock()

    tasks = []
    for im_id, im_result in results.items():
        if not im_result.get('response'):
            im = {'id': im_id, 'coco_url': im_result['url']}
            tasks.append(fetch_and_process(im, results, rate_limit_interval, semaphore, lock))
    
    for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await task  # Ensure tasks are completed

    return results

with open(target_path, 'r') as f:
    results = json.load(f)
    print('----------non-auto fill file loaded--------')

print('Reprocessing errors...')
updated_results = asyncio.run(reprocess_errors(results, rate_limit_interval=0.5))

# Save the updated results dictionary
try:
    with open(autofill_target_path, 'w') as file:
        json.dump(updated_results, file)
    print('-----------')
    print('Updated Dict Saved!')
    print('-----------')
except Exception as e:
    print(f"Error saving updated results file: {e}")

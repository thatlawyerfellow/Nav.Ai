import openai
from PyPDF2 import PdfReader
import warnings
from urllib3.exceptions import InsecureRequestWarning, NotOpenSSLWarning
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def read_pdf(file_path):
    """Read text from a PDF file."""
    print("Reading the PDF file...")
    pdf_text = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            pdf_text += page.extract_text()
    print("Finished reading the PDF file.")
    return pdf_text

def split_text_into_chunks(text, chunk_size=2000):
    """Split text into chunks of a specified size."""
    print(f"Splitting text into chunks of {chunk_size} words each...")
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])
    print("Finished splitting text into chunks.")

def generate_prompt(system_message, user_message):
    """Generate a prompt for the OpenAI API."""
    return {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    }

def call_openai_api(prompt):
    """Call the OpenAI API with the given prompt."""
    response = openai.ChatCompletion.create(
        model=prompt["model"],
        messages=prompt["messages"]
    )
    return response["choices"][0]["message"]["content"]

def main():
    # Set your OpenAI API key
    openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key

    # Specify the PDF file path
    pdf_file_path = "a.pdf"

    # Step 1: Read the PDF
    story_text = read_pdf(pdf_file_path)

    # Process the story in chunks to adhere to the token limit
    chunks = list(split_text_into_chunks(story_text))

    # Initialize a variable to track act and scene numbers
    act_number = 1
    scene_number = 1

    # Step 2: Process each chunk into acts and scenes
    system_message = "You are a helpful assistant."
    print("Processing each chunk into acts and scenes...")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")

        # Arrange into acts and scenes
        user_message = f"Arrange the following story chunk into acts and scenes. Each scene should include the setting, main characters involved, and a brief summary of the action. Here’s the story chunk: {chunk}"
        prompt = generate_prompt(system_message, user_message)
        acts_and_scenes = call_openai_api(prompt)
        
        # Convert each act into a 1000-word screenplay
        user_message = f"Convert the following act into a 1000-word screenplay with dialogue. Ensure it adheres to standard screenplay format. Here are the acts and scenes: {acts_and_scenes}"
        prompt = generate_prompt(system_message, user_message)
        formatted_scene = call_openai_api(prompt)

        # Save each act to a separate text file
        act_filename = f"act{act_number}_scene{scene_number}.txt"
        with open(act_filename, "w") as act_file:
            act_file.write(formatted_scene)
        print(f"Saved {act_filename}")

        # Increment scene number and act number if necessary
        scene_number += 1
        if scene_number > 2:  # Assuming each act can have up to 2 scenes
            act_number += 1
            scene_number = 1

    print("Finished processing all chunks into acts and scenes.")

    # Step 3: Combine all acts into a final screenplay
    print("Combining all acts into a final screenplay...")
    final_screenplay = ""
    for i in range(1, act_number + 1):
        for j in range(1, 3):
            act_filename = f"act{i}_scene{j}.txt"
            if os.path.exists(act_filename):
                with open(act_filename, "r") as act_file:
                    final_screenplay += act_file.read() + "\n\n"

    with open("output.txt", "w") as file:
        file.write(final_screenplay)
    print("Screenplay saved as output.txt")

if __name__ == "__main__":
    main()

from PIL import Image

def transform_image(image_path, to_size = 256):
    """
    Resize an image so that its width and height are at most 256 pixels,
    maintaining the aspect ratio, and replace the original image with the resized one.

    Args:
        image_path (str): Path to the image to be resized.
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get original dimensions
            original_width, original_height = img.size

            # Determine the scaling factor while maintaining the aspect ratio
            scaling_factor = min(to_size / original_width, to_size / original_height)

            # Calculate the new dimensions
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)

            # Resize the image (no need for ANTIALIAS anymore, it's default)
            resized_img = img.resize((new_width, new_height))

            # Save the resized image to the original path
            resized_img.save(image_path)

            print(f"Image resized and saved to the original path: {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_system_prompt(retrieved_memmories_text, retrieved_memmories_time):
  """
  Creates a system prompt using retrieved memories.

  Args:
    retrieved_memmories_text: A list of retrieved text memories.
    retrieved_memmories_time: A list of timestamps corresponding to the memories.

  Returns:
    A string containing the system prompt.
  """

  histor = ('; '.join([f'**{timestamp}**:\n{text}'
                  for text, timestamp in zip(retrieved_memmories_text, retrieved_memmories_time)]))

  system_prompt = (f"""
  You are an assistant that has access to a limited memory of past conversations.
  This memory includes the following, that may contain some importand preferences of User:

  {histor}

  Use this information to provide relevant and informative responses to the user's questions.
  If the memory is not relevant to the current question, disregard it.
  """)
  return system_prompt

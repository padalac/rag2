import fitz
import io
from PIL import Image
import os
from langchain.llms import OpenAI
import base64
import configparser
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
import docx2txt

min_image_height = 100
min_image_width = 100
min_image_file_size = 10000

load_dotenv(find_dotenv())


def read_config():
  config = configparser.ConfigParser()
  try:
    config.read('rag_config.ini')
    #print("mode ", config['DEFAULT']['mode'])
  except:
    print('config.ini file not found')
    exit(1)
    
  return config

rag_config = read_config()

def create_a_folder(path, folder_name):
  folder_loc = os.path.join(path, folder_name)
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_loc):
      os.makedirs(folder_loc)
  return folder_loc

def get_text_descr_from_image(image_path):
  #image_path = "Output/Images/image1.jpg"
  image_data = ""
  with open(image_path, "rb") as image_file:
    image =  base64.b64encode(image_file.read()).decode('utf-8')

  chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
  msg = chain.invoke(
      [   AIMessage(
          content="You are a helpful assistant and can describe images."
          ),
          HumanMessage(
              content=[
                  {"type": "text", "text": "Describe this image?"},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{image}"
                      },
                  },
              ]
          )
      ]
  )
  #print(msg.content)
  return msg.content

def process_images(in_file, out_dir):
  image_loc = create_a_folder(out_dir, "Images")

  fp = fitz.open(in_file)
  for page_index in range(len(fp)):
    page = fp[page_index]
    image_list = page.get_images()
    for image_index, img in enumerate(image_list, start=1):
        # get the XREF of the image
        xref = img[0]
        # extract the image bytes
        base_image = fp.extract_image(xref)
        image_bytes = base_image["image"]
        # get the image extension
        image_ext = base_image["ext"]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        if image.height < min_image_height or image.width < min_image_width:
          continue
        print(image.height, image.width)
        # save it to local disk
        img_name = f"image_{page_index+1}_{image_index}.{image_ext}"
        img_file = os.path.join(image_loc, img_name)
        image.save(open(img_file, "wb"))

  return image_loc

def get_all_image_descriptions(image_loc, text_loc):
  # Get the text description of the image and write to a text file
  # Traverse the image_loc folder and process each file
  for file_name in os.listdir(image_loc):
    img_file = os.path.join(image_loc, file_name)
    img_descr_name = f"{os.path.splitext(file_name)[0]}_descr.txt"
    text_descr = get_text_descr_from_image(img_file)
    text_file = os.path.join(text_loc, img_descr_name)
    with open(text_file, "w") as ftxt:
      ftxt.write(text_descr)
      
def process_text(in_file, out_dir, file_name):
  fp = fitz.open(in_file)
  text_loc = create_a_folder(out_dir, "Text")
  text_path = os.path.join(text_loc, f"Text_{file_name}.txt")
  with open(text_path, "wb") as fout:
    for page_index in range(len(fp)):
      page = fp[page_index]
      fout.write(page.get_text().encode("utf-8") + bytes((12,)))
  return text_loc

def del_small_files(output_folder, min_size):
  # Get list of all files only in the given directory
  func = lambda x : os.path.isfile(os.path.join(output_folder,x))
  files_list = filter(func, os.listdir(output_folder))
  for f in files_list:
    f_path = os.path.join(output_folder, f)
    if os.stat(f_path).st_size < min_size:
      print(f_path)
      os.remove(f_path)
      
def process_image_and_text_from_docx(file_name, file_path, output_folder):
  text_loc = create_a_folder(output_folder, "Text")
  image_loc = create_a_folder(output_folder, "Images")
  text = docx2txt.process(file_path, image_loc)
  #f_name = os.path.splitext(file_name)[0]
  #f_name = file_name
  text_path = os.path.join(text_loc, f"Text_{file_name}.txt")
  with open(text_path, "w") as f2:
    f2.write(text)
  del_small_files(image_loc, min_image_file_size)
  return text_loc, image_loc
  #return text
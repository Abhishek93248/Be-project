from gradio_client import Client

client = Client("https://levihsu-ootdiffusion.hf.space/--replicas/sfxdg/")
result = client.predict(
		"/content/model.jpg",	# filepath  in 'Model' Image component
		"/content/garment.jpg",	# filepath  in 'Garment' Image component
		1,	# float (numeric value between 1 and 4) in 'Images' Slider component
		20,	# float (numeric value between 20 and 40) in 'Steps' Slider component
		1,	# float (numeric value between 1.0 and 5.0) in 'Guidance scale' Slider component
		-1,	# float (numeric value between -1 and 2147483647) in 'Seed' Slider component
		api_name="/process_hd"
)


from PIL import Image
import io
# URL of the image
print(result[0]['image'])
image_url = "https://levihsu-ootdiffusion.hf.space/--replicas/sfxdg/file=" + result[0]['image']

# Send a GET request to the URL to download the image

response = requests.get(image_url)

image = Image.open(io.BytesIO(response.content))
# image = Image.open(response.raw)
# # Open the image using PIL


# # Save the image to the Colab environment
output_path = '/content/output_image.png'
image.save(output_path)

# print(f'Image saved successfully at: {output_path}')
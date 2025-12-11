
import requests

def main():
    url = "https://xrays-555583252134.europe-west1.run.app/do_you_have_covid"

    # The file you want to send
    file_path = r"C:\Users\elias\Desktop\CODE_PROJS\LeWagon\X-RAYS\x-rays-lewagon\x_rays\streamlit_frontend\test_image.png"
    files = {
        "received_image": (file_path, open(file_path, "rb"), "image/png")
    }

    response = requests.post(url, files=files)

    print("Status code:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    main()
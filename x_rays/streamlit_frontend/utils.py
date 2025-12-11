import requests


def request_prediction(uri, uploaded_file):

    # The file you want to send
    # file_path = r"C:\Users\elias\Desktop\CODE_PROJS\LeWagon\X-RAYS\x-rays-lewagon\x_rays\streamlit_frontend\test_image.png"
    files = {
        "received_image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }

    response = requests.post(uri, files=files)

    print("Status code:", response.status_code)
    print("Response:", response.text)
    
    return response
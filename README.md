# pet_social_pet_detection
### build
#### docker build pet/detection .
### run container port: 2000 5000
#### docker run -p 2000:5000 pet/detection
### request
#### Content-Type: form-data: {
    file: file
    model_choice: text
    result_type: text (json or not)
}
### response
#### Content-Type: application/json or image/jpeg
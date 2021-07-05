# pne-lung-gradio

1. Create app.py with gradio interface
2. Create requirements.txt file
3. Create setup.sh with below config
        export GRADIO_SERVER_NAME=0.0.0.0 
        export GRADIO_SERVER_PORT="$PORT"
4. Create Procfile which initiates the whole process
        web: source setup.sh && python app.py
5. Now deploy to heroku app

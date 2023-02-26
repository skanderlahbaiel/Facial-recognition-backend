from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
from resources.identify import  Identify
from resources.AddFace import  Addface

config = {
    "DEBUG": True  # run app in debug mode
}

app = Flask(__name__)

CORS(app, supports_credentials=True)
api = Api(app)


api.add_resource(Identify, '/identify')
api.add_resource(Addface, '/addface')

#api.add_resource(ImageUpload, '/image')


@app.route('/')
def helloword():
    return 'test api'


if __name__ == "__main__":
    app.run(debug=True)

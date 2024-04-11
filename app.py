import base64
import os
from flask import Flask, jsonify, make_response,request,session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import cv2
from PIL import Image, ImageOps
from numpy import array,frombuffer,float64,uint8
from classify import  classify
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
port = int(os.environ.get('PORT', 33507))
@app.route('/')
def index():
    #t=classify('img1.jpg')
    return "Method not found"
@app.route('/api/newpic',methods = ['POST'])
def newpic():
    #print('Hello')
    #print('*******',request.__dict__.keys())
    #print(request.files)
    target=os.path.join(app.root_path,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files.get('currImg')
    id = request.form.get('id')
    #print("ID-------",id)
    #print(file.__dict__.keys())
    #print(file.read())
   # print(request.form)
   # print(request.files)
    #print(request.headers['Content-Type'] )
    #print(base64.b64encode(file))
    #print("___________",request.form)
    #img=open(file,'rb')
    # SIZE = (299, 299)
    #print(file.stream.read())
    stream=file.stream.read().replace(b"data:image/jpeg;base64,",b"")
    #print(stream.decode('base64'))
    stream=base64.b64decode((stream))
    frame = cv2.imdecode(frombuffer(stream, uint8), cv2.IMREAD_COLOR)
   # print(stream)

    # with open('temp.jpeg', 'wb') as f:
    #         f.write(base64.b64decode((stream)))
    #         f.close()
    #stream=base64.b64encode(stream)
    #image = Image.open(io.BytesIO(stream))
    #image = Image.open(decoStream)
    # image = ImageOps.fit(image, SIZE)
    # image.show()
    #q = frombuffer(stream)
    #image=array(stream)
    res_cap=classify(frame)
    print(res_cap)
    # return make_response('Hi! A')
    # filename = secure_filename(file.filename)
    #destination="/".join([target, filename])
    # file.save(destination)
    #session['uploadFilePath']=destination
    res = jsonify(id=id,captions=res_cap)
    return res
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)
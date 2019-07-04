from flask_cors import CORS
from flask import request, Flask
from service import domainPredict
app=Flask(__name__)

CORS(app,supports_credentials=True)

@app.route('/domainAnalysis',methods=['GET'])
def domainAnalysisController():
    doc = request.args.get('content')
    result=domainPredict.predict(doc)
    return result
app.run(host='127.0.0.1',port=15000,debug=True)

from flask import Flask, request, render_template
import nltk
from flask_ngrok import run_with_ngrok

from model_process import predict_insincerity

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form1():
    return render_template('form.html')

@app.route('/text', methods=['GET','POST'])
def my_form_post():
    if request.method == 'POST':
        text = request.form['text1'].lower()

        output = predict_insincerity(text,64)
        output = list(output)[0]

        return render_template('form.html', final=output, text1=text)
    else:
        return render_template('form.html')

if __name__ == "__main__":
    app.run()
    #app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
# Is education really making good people nowadays? what are the simple and stylish names of mobile shops? how will you describe fascism?
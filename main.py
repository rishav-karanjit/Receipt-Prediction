from flask import Flask, render_template, jsonify
from helper.utils import createGraph
from helper.models import predictProphetModel, predictGRUModel

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictProphet', methods=['GET'])
def predictProphet():
    monthly_predictions, forecast_df = predictProphetModel()
    return jsonify({
        'data': monthly_predictions.to_dict(orient='records'),
        'graph': createGraph(forecast_df)
    })

@app.route('/predictGRU', methods=['GET'])
def predictGRU():
    monthly_predictions, forecast_df = predictGRUModel()
    return jsonify({
        'data': monthly_predictions.to_dict(orient='records'),
        'graph': createGraph(forecast_df)
    })

if __name__ == '__main__':
    app.run(debug=True)

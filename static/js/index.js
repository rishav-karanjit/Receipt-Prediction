function showDiv() {
    var dropdownValue = document.getElementById("prediction-method").value;

    // Initially hide both divs
    document.getElementById("GRU-Prediction").style.display = "none";
    document.getElementById("Prophet-Prediction").style.display = "none";

    // Show the relevant div based on dropdown selection
    if (dropdownValue == "GRU") {
        document.getElementById("GRU-Prediction").style.display = "block";
    } else if (dropdownValue == "Prophet") {
        document.getElementById("Prophet-Prediction").style.display = "block";
    }
}
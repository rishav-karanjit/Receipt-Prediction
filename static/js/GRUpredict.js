// Fetch data from the Flask backend
fetch('/predictGRU')
.then(response => response.json())
.then(data => {
    const table = document.getElementById('GRU-table');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');

    // If data is available and not empty
    // Handle the table data similar to before
    const tableData = data.data;
    if(tableData.length > 0) {
        // Add table headers
        const headers = Object.keys(tableData[0]);
        let headerRow = '<tr>';
        for(let header of headers) {
            headerRow += `<th>${header}</th>`;
        }
        headerRow += '</tr>';
        thead.innerHTML = headerRow;

        // Add table rows
        for(let row of tableData) {
            let rowHTML = '<tr>';
            for(let header of headers) {
                rowHTML += `<td>${row[header]}</td>`;
            }
            rowHTML += '</tr>';
            tbody.innerHTML += rowHTML;
        }
    } else {
        table.innerHTML = '<tr><td>No data available</td></tr>';
    }
    
    const graphData = JSON.parse(data.graph);
    Plotly.newPlot('GRUplotlyDiv', graphData.data, graphData.layout);
})
.catch(error => {
    console.error('There was an error fetching the data', error);
});
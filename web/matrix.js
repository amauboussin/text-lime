var margin = {top: 50, right: 50, bottom: 200, left: 250};


function Matrix(options) {
	    var width = options.width,
	    height = options.height,
	    data = options.data,
	    container = options.container,
	    labelsData = options.labels,
	    startColor = options.start_color,
	    endColor = options.end_color;

	var maxValue = d3.max(data, function(layer) { return d3.max(layer, function(d) { return d; }); });

  // add col and row info to each cell
  data = data.map(function(row, row_index){
	   return row.map(function(col, col_index) {
	     return {'value': data[row_index][col_index], "col": col_index, "row": row_index}
     });
  });

	var numrows = data.length;
	var numcols = data[0].length;

	var svg = d3.select(container).append("svg")
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
		.append("g")
	    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	var x = d3.scaleBand()
	    .domain(d3.range(numcols))
	    .range([0, width]);

	var y = d3.scaleBand()
	    .domain(d3.range(numrows))
	    .range([0, height]);

	var colorMap = d3.scaleLinear()
	    .domain([0, maxValue])
	    .range([startColor, endColor]);

	var x_label = svg.append("text")
		.text("True Label")
		.attr("text-anchor", "start")
		.attr("x", -2.2 * x.bandwidth())
		.attr("y", numrows / 2 * y.bandwidth() + 6)
		.attr("class", "axis-label");

	var y_label = svg.append("text")
		.text("Predicted Label")
		.attr("text-anchor", "middle")
		.attr("x", numcols / 2. * x.bandwidth())
		.attr("y", height +  y.bandwidth() + 30)
		.attr("class", "axis-label");


	var row = svg.selectAll(".row")
		.data(data)
		.enter().append("g")
		.attr("class", "row")
		.attr("transform", function(d, i) { return "translate(0," + y(i) + ")"; });

	var cell = row.selectAll(".cell")
		.data(function(d) { return d; })
		.enter().append("g")
		.attr("transform", function(d, i) { return "translate(" + x(i) + ", 0)"; });

	cell.append('rect')
    .attr("class", function(d) {return "cell " + "row" + d.row + " col" + d.col;})
    .attr("width", x.bandwidth()-2)
    .attr("height", y.bandwidth()-2)
    .style("stroke-width", "2px")
    .style("stroke", function(d){return colorMap(d.value)})
    .style("fill", function(d){return colorMap(d.value)});


  cell.append("text")
    .attr("dy", ".32em")
    .attr("x", x.bandwidth() / 2)
    .attr("y", y.bandwidth() / 2)
    .attr("text-anchor", "middle")
    .style("fill", function(d, i) { return d >= maxValue/2 ? 'white' : 'black'; })
    .text(function(d, i) { return d.value; });

  cell
    .on("mouseover", function(d) {dispatch.call("cell-highlight", this, d.row, d.col)})
    .on("mouseout", function (d) {dispatch.call("cell-unhighlight", this, d.row, d.col)})
    .on("click", function(d) {dispatch.call("cell-click", this, d.row, d.col)});

	var labels = svg.append('g')
		.attr('class', "labels");

	var columnLabels = labels.selectAll(".column-label")
	    .data(labelsData)
	    .enter().append("g")
	    .attr("class", "column-label")
	    .attr("transform", function(d, i) { return "translate(" + x(i) + "," + height + ")"; });

	// label boxes
	columnLabels.append("rect")
		.attr("class", "label-box")
		.attr("width", x.bandwidth())
		.attr("height", y.bandwidth())
    .style("stroke-width", "0px")
		.attr("fill", "#f7f7f7");

	// left dividing line
	columnLabels.append("line")
		.style("stroke", "black")
	    .style("stroke-width", ".5px")
	    .attr("x1", -.25)
	    .attr("x2", -.25)
	    .attr("y1", 0)
	    .attr("y2", y.bandwidth());

	//right dividing line
	columnLabels.append("line")
		.style("stroke", "black")
	    .style("stroke-width", ".5px")
	    .attr("x1", x.bandwidth() + .25)
	    .attr("x2", x.bandwidth() + .25)
	    .attr("y1", 0)
	    .attr("y2", y.bandwidth());

	columnLabels.append("line")
		.style("stroke", "black")
	    .style("stroke-width", "1px")
	    .attr("x1", x.bandwidth() / 2)
	    .attr("x2", x.bandwidth() / 2)
	    .attr("y1", 0)
	    .attr("y2", 5);

	columnLabels.append("text")
	    .attr("x", 15)
	    .attr("y", y.bandwidth() / 2)
	    .attr("dy", ".22em")
	    .attr("text-anchor", "end")
	    .attr("transform", "rotate(-60)")
	    .text(function(d, i) { return d; });

  columnLabels
    .on("mouseover", function(d, i) {dispatch.call("cell-highlight", this, -1, i)})
    .on("mouseout", function (d, i) {dispatch.call("cell-unhighlight", this, -1, i)})
    .on("click", function(d, i) {dispatch.call("cell-click", this, -1, i)});

	var rowLabels = labels.selectAll(".row-label")
	    .data(labelsData)
	  .enter().append("g")
	    .attr("class", "row-label")
    .attr("id", function(d, i){ return "row" + i})
			.style("fill", function(d, i) { return d >= maxValue/2 ? 'white' : 'black'; })
	    .attr("transform", function(d, i) { return "translate(" + 0 + "," + y(i) + ")"; });

		// label boxes
	rowLabels.append("rect")
		.attr("class", "label-box")
		.attr("transform", function(d, i) { return "translate(-" + x.bandwidth() + "," +0 + ")"; })
		.attr("width", x.bandwidth())
		.attr("height", y.bandwidth())
    .style("stroke-width", "0px")
		.attr("fill", "#f7f7f7");

	// top lines
	rowLabels.append("line")
		.style("stroke", "black")
	    .style("stroke-width", ".5px")
	    .attr("x1", -x.bandwidth())
	    .attr("x2", 0)
	    .attr("y1", 0)
	    .attr("y2", 0);

	// bottom lines
	rowLabels.append("line")
		.style("stroke", "black")
	    .style("stroke-width", ".5px")
	    .attr("x1", -x.bandwidth())
	    .attr("x2", 0)
	    .attr("y1", y.bandwidth())
	    .attr("y2", y.bandwidth());

	rowLabels.append("line")
		.style("stroke", "black")
	    .style("stroke-width", "1px")
	    .attr("x1", 0)
	    .attr("x2", -5)
	    .attr("y1", y.bandwidth() / 2)
	    .attr("y2", y.bandwidth() / 2);

	rowLabels.append("text")
	    .attr("x", -8)
	    .attr("y", y.bandwidth() / 2 )
	    .attr("dy", ".32em")
	    .attr("text-anchor", "end")
	    .text(function(d) { return d; });

	rowLabels
    .on("mouseover", function(d, i) {dispatch.call("cell-highlight", this, i, -1)})
    .on("mouseout", function (d, i) {dispatch.call("cell-unhighlight", this , i, -1)})
    .on("click", function(d, i) {dispatch.call("cell-click", this, i, -1)});


  // handle cell highlighting
  dispatch.on("cell-highlight", function(row, col) {
    change_border_color(row, col, true);
  });

  dispatch.on("cell-unhighlight", function(row, col) {
    change_border_color(row, col, false)
  });

  dispatch.on("cell-click.highlighting", function(row, col) {
    d3.selectAll(".cell").classed("selected", false);
    var add_selected = function(selection){
      selection.classed("selected", true)
    };
    apply_to_cells(row, col, add_selected)
  });

  //apply f to the selected cells
  function apply_to_cells(row, col, f){
    if (row === -1){
      f(d3.selectAll(".col" + col));
    }
    else if (col === -1){
      f(d3.selectAll(".row" + row));
    }
    else{
      f(d3.select(".row" + row + ".col" + col));
    }
  }

  function change_border_color(row, col, is_highlighted){
    var new_color = function(d){return is_highlighted ? "rgb(107, 107, 107)" : colorMap(d.value);};

    var toggle_color = function(selection){
      return selection.style("stroke", new_color);
    };

    apply_to_cells(row, col, toggle_color);
  }

}

// The table generation function
function tabulate(data, columns) {
    var table = d3.select("#dataView").append("table")
            .attr("style", "margin-left: " + margin.left +"px"),
        thead = table.append("thead"),
        tbody = table.append("tbody");

    // append the header row
    thead.append("tr")
        .selectAll("th")
        .data(columns)
        .enter()
        .append("th")
            .text(function(column) { return column; });

    // create a row for each object in the data
    var rows = tbody.selectAll("tr")
        .data(data)
        .enter()
        .append("tr");

    // create a cell in each row for each column
    var cells = rows.selectAll("td")
        .data(function(row) {
            return columns.map(function(column) {
                return {column: column, value: row[column]};
            });
        })
        .enter()
        .append("td")
        .attr("style", "font-family: Courier") // sets the font style
            .html(function(d) { return d.value; });

    return table;
}
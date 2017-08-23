
var bar_width = 300;
var bar_height = 300;
var bar_tick_height = 5;
var bar_chart;
var bar_transition_duration = 250;
var bar_title_offset = 30;
var bar_margin = {
  "left": 100,
  "right": 0,
  "top": 50,
  "bottom": 0
};

function make_bar_chart(container){


	var svg = container.append("svg")
	    .attr("width", bar_width + bar_margin.left + bar_margin.right)
	    .attr("height", bar_height + bar_margin.top + bar_margin.bottom)
		.append("g")
	    .attr("transform", "translate(" + bar_margin.left + "," + bar_margin.top + ")");


  var x = d3.scaleLinear()
    .domain([0, 1.05])
    .range([0, bar_width]);
  var y = d3.scaleBand()
    .range([bar_height, bar_tick_height + bar_margin.top]);

  var chart_svg = svg.append("g");

  var title = chart_svg.append("text")
    .attr("id", "bar-title")
    .attr("x", bar_width / 2)
    .attr("y", 0)
    .attr("text-anchor", "middle")
    .text("Predicted Probabilities");

  var x_axis = chart_svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + (bar_height + bar_title_offset) + ")");

  var y_axis = chart_svg.append("g")
      .attr("class", "y axis");

   bar_chart =  {
      "svg": chart_svg,
      "x": x, "x_axis": x_axis,
      "y": y, "y_axis": y_axis
   }
}

 function update_bar_chart(doc){
  // change y domain according to sorted order

   var ordered_labels = class_labels.slice().sort(function(a, b){
     return doc[class_labels.indexOf(a)] -  doc[class_labels.indexOf(b)];
   });
   bar_chart.y.domain(ordered_labels.map(function(d, i) { return d; })).padding(0.1);

    bar_chart.x_axis.transition(bar_transition_duration)
      .call(d3.axisTop(bar_chart.x)
          .ticks(6).tickFormat(function(d){
              return d; })
          .tickSizeInner([bar_height]));
    bar_chart.y_axis.transition(bar_transition_duration)
      .call(d3.axisLeft(bar_chart.y));

    var bars = bar_chart.svg.selectAll("rect")
      .data(class_labels, function(d){return d + doc["id"]});

    bars.enter().append("rect")
      .attr("class", "bar")
      .attr("height", bar_chart.y.bandwidth())
      .attr("y", function(d) { return bar_chart.y(d); })
      .attr("width", function(d, i) {return bar_chart.x(doc[i]); })
      .style("fill", function(d, i){return class_colors[i];})
      .style("stroke",  function(d, i){return class_colors[i];});
    bars.exit().remove();
    bars.transition().duration(bar_transition_duration)
      .attr("y", function(d) { return bar_chart.y(d); })
      .attr("height", bar_chart.y.bandwidth())
      .attr("width", function(d, i) {return bar_chart.x(doc[i]); });


 }
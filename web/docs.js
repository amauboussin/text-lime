


var redScale = d3.scaleLinear()
	    .range(["#ffffff", "#ff6b6b"]);

var blueScale = d3.scaleLinear()
	    .range(["#ffffff", "#5c9afd"]);

function get_docs_by_id(examples){
  var docs_by_id = {};
  for (i in examples){
    docs_by_id[examples[i]["id"]] = examples[i];
  }
  return docs_by_id
}

function get_docs_by_cell(examples){
  var docs_by_cell = [];
  for (var i=0; i<class_labels.length;i++) {
    docs_by_cell[i] = [];
    for (var j=0; j<class_labels.length; j++){
      docs_by_cell[i][j] = [];
    }
  }

  for (i in examples){
    docs_by_cell[examples[i]["label"]][examples[i]["predicted"]].push(examples[i]);
  }
  return docs_by_cell
}

function create_doc_select(){
  return d3.select("#doc-select")
    .on("change", function() { dispatch.call("doc-select",
        this, this.value); });
}

function update_doc_select(docs){
  var options = doc_select.selectAll("option")
    .data(docs, function(d){return d["id"];});

  options.enter().append("option")
    .attr("value", function (d){return d["id"];})
    .text(function (d) {
      var last_allowed_space = d["content"].substring(0, 40).lastIndexOf(" ");
      return d["content"].substring(0, last_allowed_space) + "...";
    });

  options.exit().remove();

  doc_select.property('value', docs[0]["id"]);
  update_doc_viewer(docs[0]);


  doc_select.on("change", function(d){
    update_doc_viewer(docs_by_id[doc_select.property("value")])
  });

}

function doc_max_importance(doc){
  return d3.max(Object.values(doc["explanation"]), function (val) {return Math.abs(val[1])});
}

function batch_max_importance(docs){
    //update color scales to the max values of a batch of docs
  var max_importance = d3.max(docs, doc_max_importance());
}

function create_label_select(labels){

  var predicted_select = d3.select("#predicted-label-select");
  var label_select = d3.select("#true-label-select");
  labels = ["any class"].concat(labels);
  predicted_select
    .selectAll("option")
    .data(labels)
    .enter().append("option")
    .attr("value", function(d, i){return i;})
    .text(function(d){return d;})
    .on("change", function(d, i) {dispatch.call("cell-click",-1, i)});

  label_select
    .selectAll("option")
    .data(labels)
    .enter().append("option")
    .attr("value", function(d, i){return i;})
    .text(function(d){return d;});

  var on_change = function(d) {dispatch.call("cell-click", this,
      label_select.property("value") - 1, predicted_select.property("value") - 1)};

  label_select.on("change", on_change);
  predicted_select.on("change", on_change);

}

function create_viewer(){
  return d3.select("#doc-viewer");

}

function update_doc_viewer(doc) {
  var max_importance = d3.max([.01, doc_max_importance(doc)]);
  blueScale.domain([0, max_importance]);
  redScale.domain([0, max_importance]);

  var model_correct = (doc["label"]  === doc["predicted"]);
  var positiveContrib = model_correct ? blueScale : redScale;
  var negativeContrib =  model_correct ? redScale : blueScale;

  d3.select("#doc-id")
    .text("ID: " + doc["id"]);
  d3.select("#doc-label")
    .text("Label: " + class_labels[doc["label"]]);
  d3.select("#doc-prediction")
    .text("Predicted: " + class_labels[doc["predicted"]]);

  var tokens = doc["explanation"];
  doc_viewer.selectAll(".token").remove();
  var token_divs = doc_viewer.selectAll(".token")
    .data(tokens, function(d){return doc["id"] + "|" + d;})
    .enter().append("span")
    .attr("class", "token");
  var token_text = token_divs.append("span")
    .text(function(d){return d[0];})
    .style("background", function (d) {
        var importance = d[1];
        return importance >= 0 ? positiveContrib(importance): negativeContrib(-importance);
    });
  var spaces = token_divs.append("span")
    .text(" ")
    .style("background", "white")
  }

//put each token in a spanand modify the background color
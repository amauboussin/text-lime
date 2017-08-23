

var select_char_threshold = 45;
var class_colors = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f"];

var class_scales = class_colors.map(function(c){
  return d3.scaleLinear()
        .range(["#ffffff", c]);
});

function get_docs_by_id(examples){
  var docs_by_id = {};
  var correct = 0;
  for (var i in examples){
    docs_by_id[examples[i]["id"]] = examples[i];
    if (examples[i]["predicted"] == examples[i]["label"]){
      correct += 1;
    }
  }
  accuracy = (100 * correct / examples.length).toPrecision(4);
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
  var select = d3.select("#doc-select")
    .on("change", function() {dispatch.call("doc-select",
        this, this.value); });
  return select;
}

function update_doc_select(docs){
  var options = doc_select.selectAll("option")
    .data(docs, function(d){return d["id"];});

  options.enter().append("option")
    .attr("class", "list-group-item doc-select-item")
    .attr("value", function (d){return d["id"];})
    .text(function (d) {
      var last_allowed_space = d["content"].substring(0, select_char_threshold).lastIndexOf(" ");
      return d["content"].substring(0, last_allowed_space) + " ...";
    });

  options.exit().remove();

  doc_select.property('value', docs[0]["id"]);
  update_doc_viewer(docs[0]);
  update_bar_chart(docs[0]);
  update_contrastive_examples(docs[0]);

  doc_select.on("change", function(d){
    var doc = docs_by_id[doc_select.property("value")];
    dispatch.call("doc-select", this, doc);
  });

}


function doc_max_importance(doc){
  return d3.max(Object.values(doc["explanation"]), function (token_importances) {
    var importance_vals = token_importances.slice(1).map(function(importance, i){
      return d3.max([0, importance * doc[i]]);
    });
    return d3.max(importance_vals)
  });
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
  for (var i in class_scales){
    class_scales[i].domain([0, max_importance]);
  }

  //update header
  d3.select("#doc-id")
    .text("ID: " + doc["id"]);
  d3.select("#doc-label")
    .text("Label: " + class_labels[doc["label"]])
    .style('background', class_colors[doc["label"]]);

  d3.select("#doc-prediction")
    .text("Predicted: " + class_labels[doc["predicted"]])
    .style('background', class_colors[doc["predicted"]]);


  var tokens = doc["explanation"];
  doc_viewer.selectAll(".token").remove();
  var token_divs = doc_viewer.selectAll(".token")
    .data(tokens, function(d){return doc["id"] + "|" + d;})
    .enter().append("span")
    .attr("class", "token");
  var token_text = token_divs.append("span")
    .text(function(d){return d[0];})
    .style("background", function (d) {
      var max_value = d3.max(d.slice(1));
      var max_index = d.slice(1).indexOf(max_value);
      return class_scales[max_index](max_value * doc[max_index]);
        // var importance = d[d[""]];
        // return importance >= 0 ? positiveContrib(importance): negativeContrib(-importance);
    });
  var spaces = token_divs.append("span")
    .text(" ")
    .style("background", "white")
  }

  // TODO format this in the json blob so this javascript can be deleted
function update_contrastive_examples(doc){
  d3.selectAll('#contrastive-examples div').remove();
  var available_classes = Object.keys(doc['contrastive_examples']);
  if (available_classes.length) {
  d3.select("#contrastive-examples")
    .append("div")
    .attr("class", "section-title")
    .text("Contrastive Examples");

    for (var i in available_classes) {
      var explained_class = available_classes[i];
      var token_switches = {};
      var switch_list = doc["contrastive_examples"][explained_class];
      for (var j in switch_list) {
        token_switches[switch_list[j][0]] = switch_list[j][1][1];
      }
      var class_div = d3.select('#contrastive-examples').append("div");
      class_div.append("div")
        .attr("class", "smaller-title")
        .text("Change required to change to class from " + class_labels[doc["predicted"]] + " to " + class_labels[explained_class]);
      var token_divs = class_div.selectAll('.token')
        .data(doc["explanation"])
        .enter().append("span")
        .attr("class", "token");
      var token_text = token_divs.append("span")
        .text(function (d, i) {
          return (token_switches.hasOwnProperty(i) ? token_switches[i] : d[0]);
        })
        .style("background", function (d, i) {
          return token_switches.hasOwnProperty(i) ? class_colors[explained_class] : "white";
        });
      var spaces = token_divs.append("span")
        .text(" ")
        .style("background", "white");
    }
  }
}


const width = 800, height = 500;

const svg = d3.select("#map").append("svg")
  .attr("width", width)
  .attr("height", height);

const projection = d3.geoAlbersUsa()
  .scale(1000)
  .translate([width / 2, height / 2]);

const path = d3.geoPath().projection(projection);

const tooltip = d3.select("#tooltip");


// State name mapping from FIPS code
const nameById = new Map(Object.entries({
  "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
  "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
  "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
  "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
  "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
  "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
  "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
  "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
  "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
  "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
  "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
  "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
  "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
}));

function updateInfoBox(state, count) {
  // d3.select("#state-name").text(state || "Hover over a state");
  // d3.select("#case-count").text(`Cases: ${count || ""}`);
}

Promise.all([
  d3.json("https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"),
  d3.csv("/static/processed_data/case_count.csv")
]).then(([us, data]) => {
  const statesGeo = topojson.feature(us, us.objects.states).features;

  // Clean names + values
  data.forEach(d => {
    d["Case Count"] = +d["Case Count"];
    d.Jurisdiction = d.Jurisdiction.trim();
    d.Year = +d.Year;
  });

  const maxCount = d3.max(data, d => d["Case Count"]);
  // const color = d3.scaleSequentialLog()
  //   .domain([1, maxCount])
  //   .interpolator(d3.interpolateBlues);

  let color = d3.scaleSequentialLog(d3.interpolateBlues)

  const g = svg.append("g");
  let currentData = new Map();

  const paths = g.selectAll("path")
    .data(statesGeo)
    .enter().append("path")
    .attr("stroke", "#333")
    .attr("d", path)
    .on("mouseover", function(d) {
      const state = nameById.get(d.id);
      const val = currentData.get(state) || 0;
    
      tooltip.style("display", "block")
        .html(`<strong>${state}</strong><br/>Cases: ${val}`);
    })
    .on("mousemove", function(d) {
      tooltip
        .style("left", (d3.event.pageX + 2) + "px")
        .style("top", (d3.event.pageY - 40) + "px"); // top-right
    })
    .on("mouseout", function() {
      tooltip.style("display", "none");
    });
    

  function updateMap(year) {
    const filtered = data.filter(d => d.Year === year);
    currentData = new Map(filtered.map(d => [d.Jurisdiction, d["Case Count"]]));

    const values = filtered.map(d => d["Case Count"]).filter(v => v > 0);
    const minCount = d3.min(values);
    const maxCount = d3.max(values);

    // Update color scale dynamically for that year
    color.domain([minCount, maxCount])

    paths.transition()
      .duration(300)
      .attr("fill", d => {
        const state = nameById.get(d.id);
        const val = currentData.get(state);
        return val ? color(val) : "#eee";
      });
  }

  updateMap(+document.getElementById("year-input").value);

  d3.select("#year-input").on("change", function () {
    const year = +this.value;
    if (year >= 1658 && year <= 2019) {
      updateMap(year);
    }
  });


  // Log-scaled vertical legend
  const legendSvg = d3.select("#legend");
  const legendHeight = 400;
  const legendWidth = 20;

  const colorScale = d3.scaleLog()
    .domain([1, maxCount]) // same as map scale
    .range([legendHeight, 0]);

  // Create vertical gradient
  const defs = legendSvg.append("defs");
  const gradient = defs.append("linearGradient")
    .attr("id", "gradient")
    .attr("x1", "0%").attr("x2", "0%")
    .attr("y1", "100%").attr("y2", "0%"); // vertical

  // 10 gradient stops
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    gradient.append("stop")
      .attr("offset", `${t * 100}%`)
      .attr("stop-color", d3.interpolateBlues(t));
  }

  // Draw color bar
  legendSvg.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", legendWidth)
    .attr("height", legendHeight)
    .style("fill", "url(#gradient)");

  // Add axis
  const axisScale = d3.scaleLog()
    .domain([1, maxCount])
    .range([legendHeight, 0]);

  const axis = d3.axisRight(axisScale)
    .ticks(5, "~s");

  legendSvg.append("g")
    .attr("transform", `translate(${legendWidth}, 0)`)
    .call(axis);

  
  // 1. Populate dropdown
  const states = [...new Set(data.map(d => d.Jurisdiction))].sort();
  const stateSelect = d3.select("#state-select");
  states.forEach(state => {
    stateSelect.append("option").text(state).attr("value", state);
  });

  // 2. Setup density plot
  const densitySvg = d3.select("#density-plot");
  densitySvg.append("text")
  .attr("id", "density-title")
  .attr("x", +densitySvg.attr("width") / 2)
  .attr("y", 20)
  .attr("text-anchor", "middle")
  .style("font-size", "16px")
  .style("fill", "black")
  .style("font-family", "sans-serif")
  .text("Case Count Density Over Time");

  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const dWidth = +densitySvg.attr("width") - margin.left - margin.right;
  const dHeight = +densitySvg.attr("height") - margin.top - margin.bottom;
  const gDensity = densitySvg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleLinear().range([0, dWidth]);
  const yScale = d3.scaleLinear().range([dHeight, 0]);

  const xAxis = gDensity.append("g").attr("transform", `translate(0,${dHeight})`);
  const yAxis = gDensity.append("g");

  const line = d3.line()
    .x(d => xScale(d[0]))
    .y(d => yScale(d[1]));

  const area = d3.area()
    .x(d => xScale(d[0]))
    .y0(dHeight)      // Baseline (bottom of plot)
    .y1(d => yScale(d[1]));   // Top of Curve

  // 3. Kernel Density Estimator (Gaussian)
  function kernelDensityEstimator(kernel, X) {
    return function(V) {
      return X.map(function(x) {
        return [x, d3.mean(V, v => kernel(x-v))];
      });
    };
  }

  function kernelEpanechnikov(k) {
    return function(v) {
      v = v / k;
      return Math.abs(v) <= 1 ? (0.75 * (1-v * v)) / k : 0;
    };
  }

  // 4. Update plot for selected state
  function updateDensityPlot(state) {
    const stateData = data.filter(d => d.Jurisdiction === state);
    const years = stateData.map(d => d.Year);
    const counts = stateData.map(d => d["Case Count"]);

    if (counts.length === 0) return;

    const minY = d3.min(years);
    const maxY = d3.max(years);
    const xTicks = d3.range(minY, maxY + 1, Math.ceil((maxY - minY) / 100));

    const kde = kernelDensityEstimator(kernelEpanechnikov(10), xTicks);
    const density = kde(years.flatMap((y, i) => Array(counts[i]).fill(y)));

    xScale.domain([minY, maxY]);
    yScale.domain([0, d3.max(density, d => d[1]) * 1.1]);

    xAxis.transition().call(d3.axisBottom(xScale).ticks(10).tickFormat(d3.format("d")));
    yAxis.transition().call(d3.axisLeft(yScale));

    // X-axis label
    gDensity.selectAll(".x-label").data([null]).join("text")
    .attr("class", "x-label")
    .attr("x", dWidth / 2)
    .attr("y", dHeight + 40)
    .attr("text-anchor", "middle")
    .attr("fill", "black")
    .style("font-size", "14px")
    .text("Year");

    // Y-axis label
    gDensity.selectAll(".y-label").data([null]).join("text")
    .attr("class", "y-label")
    .attr("x", -dHeight / 2)
    .attr("y", -40)
    .attr("transform", "rotate(-90)")
    .attr("text-anchor", "middle")
    .attr("fill", "black")
    .style("font-size", "14px")
    .text("Density");


    // Update or draw line
    const existing = gDensity.selectAll(".density-line").data([density]);

    existing.enter()
    .append("path")
    .attr("class", "density-line")
    .merge(existing)
    .transition()
    .attr("fill", "none")
    .attr("stroke", "#74c0fc")
    .attr("stroke-width", 2)
    .attr("d", line);

    existing.exit().remove();

    // Fill area under the curve
    const fill = gDensity.selectAll(".density-fill").data([density]);

    fill.enter()
      .append("path")
      .attr("class", "density-fill")
      .merge(fill)
      .transition()
      .attr("fill", "#74c0fc")
      .attr("opacity", 0.3)
      .attr("d", area);

    fill.exit().remove()


    d3.select("#density-title")
    .text(`Case Count Density Over Time – ${state}`);

  }

  // 5. Attach listener
  stateSelect.on("change", function () {
    const selected = this.value;
    updateDensityPlot(selected);
  });

  // 6. Initial State
  updateDensityPlot(states[0]);

});

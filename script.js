let map;
let polygon;
let coordinates = []; // Store polygon vertices
let markers = []; // Store markers for deselection

// Initialize the map
function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 20.5937, lng: 78.9629 }, // Centered on India
    zoom: 5,
  });

  // Add search functionality
  const searchBox = new google.maps.places.SearchBox(
    document.getElementById("search-bar")
  );

  // Bias the SearchBox results towards the map's viewport
  map.addListener("bounds_changed", () => {
    searchBox.setBounds(map.getBounds());
  });

  // Handle search results
  searchBox.addListener("places_changed", () => {
    const places = searchBox.getPlaces();
    if (places.length === 0) return;

    const place = places[0];
    const location = place.geometry.location;

    addCoordinate(location);
    map.panTo(location);
    map.setZoom(15);
  });

  // Add event listener for map clicks
  map.addListener("click", (event) => {
    addCoordinate(event.latLng);
  });
}

// Add coordinates to the list and map
function addCoordinate(latLng) {
  const marker = new google.maps.Marker({
    position: latLng,
    map: map,
  });

  markers.push(marker);
  coordinates.push({ lat: latLng.lat(), lng: latLng.lng() });

  drawPolygon();

  // Allow deselecting a coordinate by clicking the marker
  marker.addListener("click", () => {
    removeCoordinate(marker);
  });
}

// Remove a coordinate and update the map
function removeCoordinate(marker) {
  const index = markers.indexOf(marker);
  if (index > -1) {
    markers[index].setMap(null);
    markers.splice(index, 1);
    coordinates.splice(index, 1);

    drawPolygon();
  }
}

// Draw the polygon on the map
function drawPolygon() {
  if (polygon) {
    polygon.setMap(null);
  }

  if (coordinates.length > 0) {
    polygon = new google.maps.Polygon({
      paths: coordinates,
      map: map,
      fillColor: "#00FF00",
      fillOpacity: 0.5,
      strokeColor: "#000000",
      strokeWeight: 2,
    });
  }
}

// Function to handle coordinates saving to the backend
function handleCoordinatesWorkflow() {
  // Save coordinates to the server
  fetch("http://127.0.0.1:5000/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ coordinates }),
  })
    .then((response) => {
      if (!response.ok) throw new Error("Failed to save coordinates");
      return response.json();
    })
    .then((data) => {
      alert(data.message); // Notify user that saving was successful
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// Clear all coordinates and markers
function clearMap() {
  if (polygon) polygon.setMap(null);
  markers.forEach((marker) => marker.setMap(null));
  markers = [];
  coordinates = [];
  document.getElementById("area-display").textContent = "";
}

// Attach event listener for the "Save Coordinates" button
document
  .getElementById("search")
  .addEventListener("click", handleCoordinatesWorkflow);

// Attach event listener for the "Clear Map" button
document
  .getElementById("clear")
  .addEventListener("click", clearMap);

// Initialize the map
initMap();

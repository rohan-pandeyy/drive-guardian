document.addEventListener("DOMContentLoaded", function() {
    var overlay = document.querySelector('.overlay');
    var acceptButton = document.getElementById('acceptButton');
    var background = document.querySelector('.background');
    var image1 = document.querySelector('.image1');
    var image2 = document.querySelector('.image2');

    acceptButton.addEventListener('click', function() {
        overlay.style.display = 'none';
        background.style.filter = 'none';
        image1.classList.add('transformed');
        image2.classList.add('transformed');
    });
});

window.addEventListener('scroll', function() {
    var scrollPosition = window.scrollY;

    // Adjust the factor (20) to control the speed of movement
    var image1Transform = "translateX(-" + scrollPosition / 20 + "px)";
    var image2Transform = "translateX(" + scrollPosition / 20 + "px)";

    image1.style.transform = image1Transform;
    image2.style.transform = image2Transform;
});

// JavaScript to show team details when the "Team" button is clicked
document.addEventListener("DOMContentLoaded", function() {
    var teamButton = document.getElementById('teamButton');
    var teamContent = document.getElementById('teamContent');

    teamButton.addEventListener('click', function(event) {
        event.preventDefault(); // Prevent default link behavior
        teamContent.style.display = 'block'; // Display the team content
    });

    // JavaScript to show about section when the "About" button is clicked
    var aboutButton = document.getElementById('aboutButton');
    var aboutContent = document.getElementById('aboutContent');

    aboutButton.addEventListener('click', function(event) {
        event.preventDefault(); // Prevent default link behavior
        aboutContent.style.display = 'block'; // Display the about content
    });

    // JavaScript to close team details when the close button is clicked
    var closeTeamContentButton = document.getElementById('closeTeamContent');
    closeTeamContentButton.addEventListener('click', function() {
        teamContent.style.display = 'none';
    });

    // JavaScript to close about section when the close button is clicked
    var closeAboutContentButton = document.getElementById('closeAboutContent');
    closeAboutContentButton.addEventListener('click', function() {
        aboutContent.style.display = 'none';
    });
});

const uploadButton = document.querySelector('.gradient-button');
const fileInput = document.getElementById('fileInput');

uploadButton.addEventListener('click', () => {
  fileInput.click(); // Trigger file input click
});

fileInput.addEventListener('change', (event) => {
    const selectedFile = event.target.files[0];
    selectedFileName = selectedFile.name;
    const fileExtension = selectedFileName.split('.').pop().toLowerCase();
    if (fileExtension !== 'mp4') {
        alert("Please select an MP4 file.");
    // You might want to clear the selection or handle the error appropriately
    } else {
        console.log("Selected MP4 file name:", selectedFileName);
        fileInput.addEventListener('change', (event) => {
    const selectedFile = event.target.files[0];
    selectedFileName = selectedFile.name;
    const fileExtension = selectedFileName.split('.').pop().toLowerCase();
    if (fileExtension !== 'mp4') {
        alert("Please select an MP4 file.");
    // You might want to clear the selection or handle the error appropriately
    } else {
        console.log("Selected MP4 file name:", selectedFileName);
        window.open(`http://127.0.0.1:5000/video_stream?filename=video_india.mov&type=v`, '_blank');
    // ... your code to handle the selected MP4 file ...
    }
});
    // ... your code to handle the selected MP4 file ...
    }
});
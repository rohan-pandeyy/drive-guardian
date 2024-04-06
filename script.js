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

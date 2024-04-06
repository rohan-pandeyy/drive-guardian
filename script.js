document.addEventListener("DOMContentLoaded", function() {
    var overlay = document.querySelector('.overlay');
    var acceptButton = document.getElementById('acceptButton');
    var background = document.querySelector('.background');
    var image1 = document.querySelector('.image1');
    var image2 = document.querySelector('.image2');

    acceptButton.addEventListener('click', function() {
        overlay.style.display = 'none';
        background.style.filter = 'none';
        image1.style.transform = 'translateX(-130px) scale(1.2)';
        image2.style.transform = 'translateX(130px) scale(1.2)';
    });
});
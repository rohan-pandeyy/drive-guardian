document.addEventListener('DOMContentLoaded', function() {
    // Get references to the background and car layers
    const backgroundLayer = document.querySelector('.background-layer');
    const carLayer = document.querySelector('.car-layer');

    // Function to update the position of the background layers based on scroll position
    function updateParallax() {
        // Calculate the scroll position relative to the window height
        const scrollPosition = window.scrollY;
        const windowHeight = window.innerHeight;
        const parallaxValue = scrollPosition / windowHeight;

        // Adjust the position of the background layers based on the scroll position
        backgroundLayer.style.transform = translateY(-$(parallaxValue * 50));
        carLayer.style.transform = translateY(-$(parallaxValue * 10)); // Adjust the parallax effect for the car layer
    }
    
    // Add event listener for scrolling to update the parallax effect
    window.addEventListener('scroll', updateParallax);

    // Call the updateParallax function initially to set the initial position
    updateParallax();
});
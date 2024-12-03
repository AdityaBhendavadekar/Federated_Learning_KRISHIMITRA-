document.getElementById("contactForm").addEventListener("submit", function(event) {
    event.preventDefault();  // Prevent form submission to avoid page refresh
    
    // Collect form data
    let parms = {
        name: document.getElementById("name").value,
        email: document.getElementById("email").value,
        phone: document.getElementById("phone").value,
        message: document.getElementById("message").value,
    };

    // Send email using EmailJS
    emailjs.send("service_bz1t63h", "template_n37n12s", parms)
        .then(function(response) {
            alert("Email Sent Successfully!!");
        })
        .catch(function(error) {
            alert("Failed to send email: " + error.text);
        });
});

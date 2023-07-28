$(function() {
    // Configure Autocomplete for the stock input field
    $('#stock-input').autocomplete({
      source: function(request, response) {
        $.ajax({
          url: '/stock_names',
          data: { query: request.term },
          dataType: 'json',
          success: function(data) {
            response(data.stock_names); // Pass the stock names to the Autocomplete widget
          }
        });
      },
      minLength: 2 // Minimum characters to trigger autocomplete
    });
  });
  
  
  // sign in and sigh up form
  const express = require('express');
  const app = express();
  const bodyParser = require('body-parser');
  
  // Middleware to parse request body
  app.use(bodyParser.urlencoded({ extended: false }));
  app.use(bodyParser.json());
  
  // Endpoint for user sign in
  app.post('/signin', (req, res) => {
    const username = req.body.username;
    const password = req.body.password;
  
    // Validate username and password
    // Example: check if they match with a user in the database
  
    // Assuming the login is valid, redirect to the home page
    res.redirect('/index.html');
  });
  
  // Endpoint for user sign up
  app.post('/signup', (req, res) => {
    const username = req.body.username;
    const email = req.body.email;
    const password = req.body.password;
  
    // Validate username, email, and password
    // Example: check if they meet certain criteria and if the username/email is available
  
    // Create new user account in the database
    // Example: store the user details in a database
  
    // Assuming the sign-up is successful, redirect to the home page
    res.redirect('/index.html');
  });
  
  // Start the server
  const port = 3000;
  app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
  });
  
  
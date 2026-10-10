// =========================================================
// AIRLINK - FLIGHT PRICE PREDICTION
// =========================================================


// =========================================================
// AIRPORT DATABASE
// =========================================================

const airports = [

    {
        city: "Delhi",
        airport: "Indira Gandhi International Airport",
        code: "DEL"
    },

    {
        city: "Mumbai",
        airport: "Chhatrapati Shivaji Maharaj International Airport",
        code: "BOM"
    },

    {
        city: "Bangalore",
        airport: "Kempegowda International Airport",
        code: "BLR"
    },

    {
        city: "Hyderabad",
        airport: "Rajiv Gandhi International Airport",
        code: "HYD"
    },

    {
        city: "Chennai",
        airport: "Chennai International Airport",
        code: "MAA"
    },

    {
        city: "Kolkata",
        airport: "Netaji Subhas Chandra Bose International Airport",
        code: "CCU"
    },

    {
        city: "Ahmedabad",
        airport: "Sardar Vallabhbhai Patel International Airport",
        code: "AMD"
    },

    {
        city: "Pune",
        airport: "Pune Airport",
        code: "PNQ"
    },

    {
        city: "Goa",
        airport: "Goa International Airport",
        code: "GOI"
    },

    {
        city: "Jaipur",
        airport: "Jaipur International Airport",
        code: "JAI"
    },

    {
        city: "Lucknow",
        airport: "Chaudhary Charan Singh Airport",
        code: "LKO"
    },

    {
        city: "Kochi",
        airport: "Cochin International Airport",
        code: "COK"
    },

    {
        city: "Thiruvananthapuram",
        airport: "Trivandrum International Airport",
        code: "TRV"
    },

    {
        city: "Coimbatore",
        airport: "Coimbatore International Airport",
        code: "CJB"
    },

    {
        city: "Madurai",
        airport: "Madurai Airport",
        code: "IXM"
    },

    {
        city: "Nagpur",
        airport: "Dr. Babasaheb Ambedkar International Airport",
        code: "NAG"
    },

    {
        city: "Indore",
        airport: "Devi Ahilya Bai Holkar Airport",
        code: "IDR"
    },

    {
        city: "Bhopal",
        airport: "Raja Bhoj Airport",
        code: "BHO"
    },

    {
        city: "Raipur",
        airport: "Swami Vivekananda Airport",
        code: "RPR"
    },

    {
        city: "Patna",
        airport: "Jay Prakash Narayan Airport",
        code: "PAT"
    },

    {
        city: "Guwahati",
        airport: "Lokpriya Gopinath Bordoloi Airport",
        code: "GAU"
    },

    {
        city: "Vadodara",
        airport: "Vadodara Airport",
        code: "BDQ"
    },

    {
        city: "Visakhapatnam",
        airport: "Visakhapatnam Airport",
        code: "VTZ"
    },

    {
        city: "Tirupati",
        airport: "Tirupati Airport",
        code: "TIR"
    },

    {
        city: "Vijayawada",
        airport: "Vijayawada Airport",
        code: "VGA"
    },

    {
        city: "Varanasi",
        airport: "Lal Bahadur Shastri Airport",
        code: "VNS"
    }

];


// =========================================================
// AUTOCOMPLETE
// =========================================================

function setupAutocomplete(inputId, resultId) {

    const input = document.getElementById(inputId);
    const results = document.getElementById(resultId);

    if (!input || !results) {
        return;
    }


    input.addEventListener("input", function () {

        const value = input.value.trim().toLowerCase();

        results.innerHTML = "";

        if (value.length === 0) {
            return;
        }


        airports.forEach(function (airport) {

            const searchableText =
                airport.city +
                " " +
                airport.airport +
                " " +
                airport.code;


            if (
                searchableText
                    .toLowerCase()
                    .includes(value)
            ) {

                const div = document.createElement("div");

                div.classList.add("result-item");


                div.innerHTML = `

                    <span class="airport-icon">
                        ✈
                    </span>

                    <div>

                        <div class="airport-name">
                            ${airport.city} - ${airport.airport}
                        </div>

                        <div class="airport-code">
                            ${airport.code}
                        </div>

                    </div>

                `;


                div.addEventListener("click", function () {

                    input.value =
                        airport.city +
                        " - " +
                        airport.code;

                    results.innerHTML = "";

                });


                results.appendChild(div);

            }

        });

    });


    // Close autocomplete when clicking outside

    document.addEventListener("click", function (event) {

        if (
            !input.contains(event.target) &&
            !results.contains(event.target)
        ) {

            results.innerHTML = "";

        }

    });

}


// =========================================================
// INITIALIZATION
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        setupAutocomplete(
            "Source",
            "sourceResults"
        );


        setupAutocomplete(
            "Destination",
            "destinationResults"
        );


        const form =
            document.getElementById(
                "predictionForm"
            );


        if (form) {

            form.addEventListener(
                "submit",
                function (event) {

                    event.preventDefault();

                    predictPrice();

                }
            );

        }


        // Set minimum travel date to today

        const dateInput =
            document.getElementById(
                "travel_date"
            );


        if (dateInput) {

            const today =
                new Date()
                    .toISOString()
                    .split("T")[0];

            dateInput.min = today;

        }

    }
);


// =========================================================
// PREDICT PRICE
// =========================================================

function predictPrice() {

    const sourceField =
        document.getElementById("Source");


    const destinationField =
        document.getElementById("Destination");


    const airlineField =
        document.getElementById("Airline");


    const dateField =
        document.getElementById("travel_date");


    if (
        !sourceField ||
        !destinationField ||
        !airlineField ||
        !dateField
    ) {

        return;

    }


    // Get values

    const source =
        sourceField.value
            .split("-")[0]
            .trim();


    const destination =
        destinationField.value
            .split("-")[0]
            .trim();


    const airline =
        airlineField.value;


    const travelDate =
        dateField.value;


    // =====================================================
    // VALIDATION
    // =====================================================

    if (!source || !destination) {

        showMessage(
            "Please select valid departure and destination airports."
        );

        return;

    }


    if (source === destination) {

        showMessage(
            "Source and destination cannot be the same."
        );

        return;

    }


    if (!travelDate) {

        showMessage(
            "Please select a travel date."
        );

        return;

    }


    // =====================================================
    // DATE VALIDATION
    // =====================================================

    const selectedDate =
        new Date(
            travelDate + "T00:00:00"
        );


    const today =
        new Date();


    today.setHours(0, 0, 0, 0);


    const timeDifference =
        selectedDate.getTime() -
        today.getTime();


    const daysBefore =
        Math.ceil(
            timeDifference /
            (1000 * 60 * 60 * 24)
        );


    if (daysBefore < 0) {

        showMessage(
            "Travel date must be today or a future date."
        );

        return;

    }


    // =====================================================
    // REQUEST DATA
    // =====================================================

    const data = {

        Airline: airline,

        Source: source,

        Destination: destination,

        Total_Stops: 1,

        Duration_Hours: 2,

        Duration_Minutes: 0,

        Journey_Month:
            selectedDate.getMonth() + 1,

        Journey_Day:
            selectedDate.getDate(),

        Days_Before:
            daysBefore

    };


    // =====================================================
    // BUTTON LOADING STATE
    // =====================================================

    const button =
        document.querySelector(
            ".search-btn"
        );


    const originalButtonHTML =
        button
            ? button.innerHTML
            : "";


    if (button) {

        button.disabled = true;

        button.innerHTML = `
            <span>⏳</span>
            <span>Analyzing...</span>
        `;

    }


    // =====================================================
    // SEND REQUEST TO FLASK
    // =====================================================

    fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type":
                "application/json"
        },

        body:
            JSON.stringify(data)

    })


    .then(function (response) {

        if (!response.ok) {

            throw new Error(
                "Server returned an error."
            );

        }

        return response.json();

    })


    .then(function (result) {


        // =================================================
        // NO DIRECT FLIGHT
        // =================================================

        if (
            result.route_exists === false
        ) {

            window.location.href =
                "/flights?source=" +
                encodeURIComponent(source) +
                "&destination=" +
                encodeURIComponent(destination);

            return;

        }


        // =================================================
        // STORE PREDICTION
        // =================================================

        sessionStorage.setItem(
            "predictionData",
            JSON.stringify(result)
        );


        sessionStorage.setItem(
            "lastSource",
            source
        );


        sessionStorage.setItem(
            "lastDestination",
            destination
        );


        sessionStorage.setItem(
            "lastAirline",
            airline
        );


        sessionStorage.setItem(
            "travelDate",
            travelDate
        );


        // =================================================
        // GO TO LOADING PAGE
        // =================================================

        window.location.href =
            "/loading";

    })


    .catch(function (error) {

        console.error(
            "Prediction error:",
            error
        );


        showMessage(
            "Prediction failed. Please try again."
        );


        // Restore button

        if (button) {

            button.disabled = false;

            button.innerHTML =
                originalButtonHTML;

        }

    });

}


// =========================================================
// MESSAGE HELPER
// =========================================================

function showMessage(message) {

    alert(message);

}
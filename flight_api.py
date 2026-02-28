"""
flight_api.py - Fetches and normalizes flight data from BudgetAir compare API.

Real API response structure (from Postman):
{
  "onwardFlightCompareResponse": [
    {
      "flightCode": "QP1831",
      "airline": "QP",
      "from": "DEL", "fromCity": "New Delhi",
      "to": "NMI",   "toCity": "Mumbai",
      "depDate": "2026-03-25", "depTime": "06:30",
      "arrDate": "2026-03-25", "arrTime": "08:40",
      "duration": "02:10",
      "stops": 0,
      "cheapestFare": 4501,
      "cheapestProvider": { "providerCode": "EASEMYTRIP" },
      "isRefundable": true,
      "seatingClass": "E",
      "checkinBaggage": ["15 Kgs"],
      "cabinBaggage": ["7 Kgs"],
      "transitFlight": [{ "viaAirportCode": "NON-STOP" }],
      "compare": {
        "EASEMYTRIP": { "fare": { "totalFare": 4501, "baseFare": 3489, "totalTax": 1012 } },
        "AERTRIP":    { "fare": { "totalFare": 4641 } }
      }
    },
    ...
  ]
}

Airline codes seen in real data:
  QP = Akasa Air, IX = Air India Express, 6E = IndiGo,
  SG = SpiceJet, UK = Vistara, AI = Air India, G8 = Go First
"""

import json
import argparse
import requests
from typing import Optional

FLIGHT_API_URL = "https://budgetair.tripsaverz.in/v1/flights/compare"

AIRLINE_NAMES = {
    "QP": "Akasa Air",
    "IX": "Air India Express",
    "6E": "IndiGo",
    "SG": "SpiceJet",
    "UK": "Vistara",
    "AI": "Air India",
    "G8": "Go First",
    "EY": "Etihad Airways",
    "EK": "Emirates",
    "LH": "Lufthansa",
}


def fetch_flights(
    origin: str = "DEL",
    destination: str = "BOM",
    doj: str = "25032026",
    adults: int = 1,
    child: int = 0,
    infant: int = 0,
    roundtrip: bool = False,
    seating_class: str = "ECONOMY",
) -> dict:
    """Hit the BudgetAir compare API and return raw JSON response."""
    payload = {
        "from": origin,
        "to": destination,
        "doj": doj,
        "adults": adults,
        "child": child,
        "infant": infant,
        "roundtrip": roundtrip,
        "seatingClass": seating_class,
    }

    print(f"[flight_api] Fetching: {origin} → {destination} on {doj}")

    try:
        response = requests.post(
            FLIGHT_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        print(f"[flight_api] ✓ API responded. Status: {response.status_code}")
        return data

    except requests.exceptions.ConnectionError:
        print("[flight_api] ⚠ Cannot reach API. Load data via --file instead.")
        return {"onwardFlightCompareResponse": []}

    except Exception as e:
        print(f"[flight_api] ⚠ Error: {e}")
        return {"onwardFlightCompareResponse": []}


def normalize_flights(api_response: dict, origin: str = "DEL", destination: str = "BOM") -> list[dict]:
    """
    Normalize the real BudgetAir API response into flat flight documents.

    Reads from: api_response["onwardFlightCompareResponse"]
    Each item becomes one document for ChromaDB embedding.
    """
    raw_flights = api_response.get("onwardFlightCompareResponse", [])

    if not raw_flights:
        print("[flight_api] ⚠ No flights in onwardFlightCompareResponse.")
        return []

    flights = []
    for flight in raw_flights:
        try:
            # Core fields directly on each flight object
            flight_code = flight.get("flightCode", "")
            airline_code = flight.get("airline", "")
            airline_name = AIRLINE_NAMES.get(airline_code, airline_code)

            # For multi-leg flights like "6E2345->6E5678", extract first leg code
            first_leg = flight_code.split("->")[0] if "->" in flight_code else flight_code

            # Route
            from_code = flight.get("from", origin)
            from_city = flight.get("fromCity", origin)
            to_code = flight.get("to", destination)
            to_city = flight.get("toCity", destination)

            # Timing
            dep_date = flight.get("depDate", "")
            dep_time = flight.get("depTime", "")
            arr_date = flight.get("arrDate", "")
            arr_time = flight.get("arrTime", "")
            duration = flight.get("duration", "")
            stops = flight.get("stops", 0)

            # Pricing — use cheapestFare as the headline price
            cheapest_fare = flight.get("cheapestFare", 0)
            cheapest_provider = flight.get("cheapestProvider", {}).get("providerCode", "")

            # Extract per-provider fares from compare dict
            compare = flight.get("compare", {})
            provider_fares = {}
            for provider, data in compare.items():
                fare = data.get("fare", {})
                total = fare.get("totalFare") or fare.get("totalFareAfterDiscount") or 0
                base = fare.get("baseFare", 0)
                tax = fare.get("totalTax", 0)
                if total:
                    provider_fares[provider] = {
                        "total": total,
                        "base": base,
                        "tax": tax,
                        "booking_url": data.get("redirecUrl", ""),
                    }

            # Transit / stops info
            transit_list = flight.get("transitFlight", [])
            via_cities = [
                t.get("viaCity") for t in transit_list
                if t.get("viaCity") and t.get("viaCity") != "NON-STOP"
            ]
            via_text = ", ".join(via_cities) if via_cities else "Non-stop"

            # Baggage
            checkin_baggage = flight.get("checkinBaggage", ["15 Kgs"])
            cabin_baggage = flight.get("cabinBaggage", ["7 Kgs"])
            checkin_text = checkin_baggage[0] if checkin_baggage else "15 Kgs"
            cabin_text = cabin_baggage[0] if cabin_baggage else "7 Kgs"

            # Refundable
            is_refundable = flight.get("isRefundable", False)

            # Seating class
            seating_map = {"E": "Economy", "B": "Business", "F": "First", "P": "Premium Economy"}
            seat_class = seating_map.get(flight.get("seatingClass", "E"), "Economy")

            # Journey type
            journey_type = api_response.get("journeyType", "ONE_WAY")
            flight_type = api_response.get("flightType", "DOMESTIC")

            doc = {
                "flight_id": flight_code,
                "flight_code": first_leg,
                "airline_code": airline_code,
                "airline": airline_name,
                "from": from_code,
                "from_city": from_city,
                "to": to_code,
                "to_city": to_city,
                "dep_date": dep_date,
                "dep_time": dep_time,
                "arr_date": arr_date,
                "arr_time": arr_time,
                "duration": duration,
                "stops": stops,
                "via": via_text,
                "cheapest_fare": cheapest_fare,
                "cheapest_provider": cheapest_provider,
                "provider_fares": json.dumps(provider_fares),  # stored as JSON string
                "checkin_baggage": checkin_text,
                "cabin_baggage": cabin_text,
                "is_refundable": is_refundable,
                "seating_class": seat_class,
                "journey_type": journey_type,
                "flight_type": flight_type,
            }

            doc["text_chunk"] = build_text_chunk(doc, provider_fares)
            flights.append(doc)

        except Exception as e:
            print(f"[flight_api] ⚠ Skipping flight {flight.get('flightCode','?')}: {e}")
            continue

    print(f"[flight_api] ✓ Normalized {len(flights)} flights.")
    return flights


def build_text_chunk(doc: dict, provider_fares: dict) -> str:
    """
    Build a natural language description of a flight for embedding.
    This is what gets semantically searched.
    """
    stops_text = "non-stop" if doc["stops"] == 0 else f"{doc['stops']} stop(s) via {doc['via']}"
    refund_text = "refundable" if doc["is_refundable"] else "non-refundable"

    # Build provider price comparison text
    price_parts = []
    for provider, fare in provider_fares.items():
        price_parts.append(f"{provider}: ₹{fare['total']} (base ₹{fare['base']} + tax ₹{fare['tax']})")
    prices_text = " | ".join(price_parts) if price_parts else f"₹{doc['cheapest_fare']}"

    chunk = (
        f"{doc['airline']} ({doc['airline_code']}) flight {doc['flight_code']} "
        f"from {doc['from_city']} ({doc['from']}) to {doc['to_city']} ({doc['to']}). "
        f"Departure: {doc['dep_date']} at {doc['dep_time']}, "
        f"Arrival: {doc['arr_date']} at {doc['arr_time']}. "
        f"Duration: {doc['duration']}, {stops_text}. "
        f"Seating class: {doc['seating_class']}. "
        f"Cheapest fare: ₹{doc['cheapest_fare']} via {doc['cheapest_provider']}. "
        f"Prices across providers — {prices_text}. "
        f"Check-in baggage: {doc['checkin_baggage']}, Cabin baggage: {doc['cabin_baggage']}. "
        f"This flight is {refund_text}. "
        f"Flight type: {doc['flight_type']}, Journey: {doc['journey_type']}."
    )
    return chunk


def load_from_json(filepath: str) -> dict:
    """Load saved Postman/API response from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    count = len(data.get("onwardFlightCompareResponse", []))
    print(f"[flight_api] ✓ Loaded {count} flights from {filepath}")
    return data


def save_to_json(data: dict, filepath: str = "flight_response.json"):
    """Save API response to JSON (mirrors what Postman exports)."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[flight_api] ✓ Saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BudgetAir flight fetcher")
    parser.add_argument("--from", dest="origin", default="DEL")
    parser.add_argument("--to", dest="destination", default="BOM")
    parser.add_argument("--date", default="25032026")
    parser.add_argument("--file", help="Load from saved JSON instead of hitting API")
    args = parser.parse_args()

    if args.file:
        raw = load_from_json(args.file)
    else:
        raw = fetch_flights(args.origin, args.destination, args.date)

    flights = normalize_flights(raw, args.origin, args.destination)
    print(f"\nTotal flights: {len(flights)}")
    for f in flights:
        print(f"  → {f['airline']} {f['flight_code']} | {f['dep_time']}→{f['arr_time']} | ₹{f['cheapest_fare']} | {f['stops']} stop(s) | {f['via']}")

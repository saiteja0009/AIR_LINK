import unittest
from datetime import datetime, timedelta
import json
from app import app, get_db

class TestTicketCancellation(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'airlink_super_secret_key_2026'
        self.client = app.test_client()

        # Insert test users and bookings
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM users WHERE username IN ('user1', 'user2')")
        cursor.execute("DELETE FROM bookings WHERE booking_id LIKE 'TEST%'")

        cursor.execute("INSERT INTO users (username, password) VALUES ('user1', 'pass123')")
        cursor.execute("INSERT INTO users (username, password) VALUES ('user2', 'pass123')")

        future_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        past_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        # Active confirmed booking for user1
        cursor.execute("""
            INSERT INTO bookings (booking_id, passenger, source, destination, airline, price, seat, pnr, date, username, booking_status)
            VALUES ('TEST101', 'Rahul Sharma', 'Hyderabad', 'Delhi', 'IndiGo', 6250.0, '12A', 'PNR101', ?, 'user1', 'CONFIRMED')
        """, (future_date,))

        # Past flight booking for user1
        cursor.execute("""
            INSERT INTO bookings (booking_id, passenger, source, destination, airline, price, seat, pnr, date, username, booking_status)
            VALUES ('TEST102', 'Rahul Sharma', 'Mumbai', 'Bangalore', 'Air India', 5000.0, '14B', 'PNR102', ?, 'user1', 'CONFIRMED')
        """, (past_date,))

        # Booking owned by user2
        cursor.execute("""
            INSERT INTO bookings (booking_id, passenger, source, destination, airline, price, seat, pnr, date, username, booking_status)
            VALUES ('TEST103', 'Priya Singh', 'Delhi', 'Chennai', 'Vistara', 8000.0, '18C', 'PNR103', ?, 'user2', 'CONFIRMED')
        """, (future_date,))

        conn.commit()
        conn.close()

    def test_01_confirmed_booking_cancellation_success(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'user1'

        res = self.client.post('/booking/TEST101/cancel', data=json.dumps({"reason": "Personal emergency"}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['original_fare'], 6250.0)
        self.assertEqual(data['cancellation_fee'], 625.0) # 10% of 6250
        self.assertEqual(data['refund_amount'], 5625.0)
        self.assertEqual(data['refund_status'], 'SIMULATED')

    def test_02_duplicate_cancellation_rejected(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'user1'

        # First cancellation
        self.client.post('/booking/TEST101/cancel', content_type='application/json')
        # Second cancellation attempt
        res = self.client.post('/booking/TEST101/cancel', content_type='application/json')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn("already been cancelled", data['message'])

    def test_03_unauthorized_user_cancellation_rejected(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'user1'

        # User1 attempts to cancel User2's booking TEST103
        res = self.client.post('/booking/TEST103/cancel', content_type='application/json')
        self.assertEqual(res.status_code, 403)
        data = res.get_json()
        self.assertEqual(data['status'], 'error')

    def test_04_expired_flight_cancellation_rejected(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'user1'

        # Attempt to cancel past flight TEST102
        res = self.client.post('/booking/TEST102/cancel', content_type='application/json')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("date has already passed", data['message'])

    def test_05_invalid_booking_id(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'user1'

        res = self.client.post('/booking/INVALID999/cancel', content_type='application/json')
        self.assertEqual(res.status_code, 404)

    def test_06_unauthenticated_user_rejected(self):
        res = self.client.post('/booking/TEST101/cancel', content_type='application/json')
        self.assertEqual(res.status_code, 401)

    def test_07_pdf_download_for_active_and_cancelled(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'user1'

        res1 = self.client.get('/download-ticket/TEST101')
        self.assertEqual(res1.status_code, 200)

        # Cancel then download ticket again
        self.client.post('/booking/TEST101/cancel', content_type='application/json')
        res2 = self.client.get('/download-ticket/TEST101')
        self.assertEqual(res2.status_code, 200)

    def test_08_admin_analytics_view(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'admin'

        res = self.client.get('/admin')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Flight Booking', res.data)

if __name__ == '__main__':
    unittest.main()

import unittest
from datetime import datetime, timedelta
import json
from app import app, get_db

class TestAIRLINKPlatform(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'airlink_super_secret_key_2026'
        self.client = app.test_client()

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM users WHERE username IN ('cust1', 'cust2', 'admin_test')")
        cursor.execute("DELETE FROM bookings WHERE booking_id LIKE 'PLAT%'")
        cursor.execute("DELETE FROM price_alerts WHERE username IN ('cust1', 'cust2')")
        cursor.execute("DELETE FROM saved_flights WHERE username IN ('cust1', 'cust2')")
        cursor.execute("DELETE FROM password_reset_otps WHERE email LIKE 'cust1%'")

        # Create test users
        from werkzeug.security import generate_password_hash
        cursor.execute("INSERT INTO users (username, email, password, role, status) VALUES ('cust1', 'cust1@gmail.com', ?, 'customer', 'active')",
                       (generate_password_hash('pass123'),))
        cursor.execute("INSERT INTO users (username, email, password, role, status) VALUES ('cust2', 'cust2@gmail.com', ?, 'customer', 'active')",
                       (generate_password_hash('pass123'),))
        cursor.execute("INSERT INTO users (username, email, password, role, status) VALUES ('admin_test', 'admin_test@airlink.com', ?, 'admin', 'active')",
                       (generate_password_hash('admin123'),))

        future_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

        cursor.execute("""
            INSERT INTO bookings (booking_id, passenger, source, destination, airline, price, seat, pnr, date, username, booking_status)
            VALUES ('PLAT101', 'Test Customer', 'Hyderabad', 'Delhi', 'IndiGo', 6250.0, '12A', 'PNRPLAT1', ?, 'cust1', 'CONFIRMED')
        """, (future_date,))

        conn.commit()
        conn.close()

    def test_01_customer_auth_and_rbac(self):
        # Customer Login
        res = self.client.post('/login', data={'username': 'cust1', 'password': 'pass123'}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        # Customer attempts to access Admin dashboard -> 403 Forbidden
        with self.client.session_transaction() as sess:
            sess['user'] = 'cust1'
            sess['role'] = 'customer'
        res_admin = self.client.get('/admin')
        self.assertEqual(res_admin.status_code, 403)

    def test_02_admin_auth_and_access(self):
        # Admin Login
        res = self.client.post('/admin-login', data={'username': 'admin_test', 'password': 'admin123'}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        with self.client.session_transaction() as sess:
            sess['user'] = 'admin_test'
            sess['role'] = 'admin'
        res_admin = self.client.get('/admin')
        self.assertEqual(res_admin.status_code, 200)

    def test_03_forgot_password_and_otp_flow(self):
        # Request OTP
        res = self.client.post('/forgot-password', data={'email': 'cust1@gmail.com'})
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'OTP', res.data)

        # Retrieve OTP from DB
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT otp FROM password_reset_otps WHERE email='cust1@gmail.com' ORDER BY id DESC LIMIT 1")
        otp = cursor.fetchone()['otp']
        conn.close()

        # Verify OTP
        with self.client.session_transaction() as sess:
            sess['reset_email'] = 'cust1@gmail.com'
            sess['reset_username'] = 'cust1'

        res_v = self.client.post('/verify-otp', data={'otp': otp}, follow_redirects=True)
        self.assertEqual(res_v.status_code, 200)

        # Reset Password
        res_r = self.client.post('/reset-password', data={'password': 'newpass123', 'confirm_password': 'newpass123'}, follow_redirects=True)
        self.assertEqual(res_r.status_code, 200)

    def test_04_flight_rescheduling(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'cust1'
            sess['role'] = 'customer'

        new_date = (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
        res = self.client.post('/booking/PLAT101/reschedule', data=json.dumps({"new_date": new_date}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['new_date'], new_date)

    def test_05_price_alerts_and_saved_flights(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'cust1'
            sess['role'] = 'customer'

        # Price Alert
        res_alert = self.client.post('/api/price-alerts/create', data=json.dumps({
            "source": "Delhi", "destination": "Mumbai", "travel_date": "2026-11-01", "target_price": 5000
        }), content_type='application/json')
        self.assertEqual(res_alert.status_code, 200)

        # Saved Flight
        res_save = self.client.post('/api/saved-flights/toggle', data=json.dumps({
            "airline": "IndiGo", "source": "Delhi", "destination": "Mumbai", "price": 5200, "time": "08:00 AM"
        }), content_type='application/json')
        self.assertEqual(res_save.status_code, 200)
        self.assertTrue(res_save.get_json()['saved'])

    def test_06_ai_assistant_privacy(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'cust1'
            sess['role'] = 'customer'

        res = self.client.post('/api/ai-assistant/chat', data=json.dumps({"message": "show my bookings"}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        self.assertIn("PLAT101", res.get_json()['reply'])

    def test_07_ml_analytics_metrics(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'admin_test'
            sess['role'] = 'admin'

        res = self.client.get('/admin/ml-analytics')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'XGBoost Regressor', res.data)

if __name__ == '__main__':
    unittest.main()

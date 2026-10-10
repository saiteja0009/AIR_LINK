import unittest
from app import app, get_db
import sqlite3

class TestAirLinkRBAC(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # Seed test customer
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username='testcustomer'")
        cursor.execute(
            "INSERT INTO users (username, email, password, role, status, created_at) VALUES (?,?,?,?,?,?)",
            ('testcustomer', 'customer@test.com', 'cust123', 'customer', 'active', '2026-09-22 00:00:00')
        )
        conn.commit()
        conn.close()

    def test_01_common_login_page_renders(self):
        """Test common login page contains both USER and ADMIN toggle options."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("USER LOGIN", html)
        self.assertIn("ADMIN LOGIN", html)
        self.assertIn('login_mode', html)

    def test_02_customer_login_success(self):
        """Test customer logs in via USER LOGIN mode and redirects to /home."""
        response = self.client.post('/login', data={
            'login_mode': 'user',
            'username': 'testcustomer',
            'password': 'cust123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("WELCOME BACK", html)
        # Ensure NO Analytics links exist in customer view
        self.assertNotIn('<a href="/analytics"', html)
        self.assertNotIn('Travel Analytics', html)

    def test_03_customer_denied_admin_mode_login(self):
        """Test customer attempting ADMIN LOGIN mode is rejected."""
        response = self.client.post('/login', data={
            'login_mode': 'admin',
            'username': 'testcustomer',
            'password': 'cust123'
        })
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("This account does not have administrator privileges", html)

    def test_04_customer_blocked_from_admin_routes(self):
        """Test logged in customer receiving 403 on /admin routes."""
        # Log in customer
        self.client.post('/login', data={'login_mode': 'user', 'username': 'testcustomer', 'password': 'cust123'})

        for route in ['/admin', '/admin/dashboard', '/admin/analytics', '/admin/users', '/admin/flights', '/admin/ml-analytics']:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 403, f"Expected 403 for customer on {route}")

    def test_05_admin_login_success(self):
        """Test admin credentials under ADMIN LOGIN mode redirect to admin dashboard."""
        response = self.client.post('/login', data={
            'login_mode': 'admin',
            'username': 'admin',
            'password': 'air123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Admin Dashboard", html)

    def test_06_admin_analytics_and_ml_analytics(self):
        """Test admin can access /admin/analytics and /admin/ml-analytics."""
        self.client.post('/login', data={'login_mode': 'admin', 'username': 'admin', 'password': 'air123'})

        # Test Admin Analytics
        res_analytics = self.client.get('/admin/analytics')
        self.assertEqual(res_analytics.status_code, 200)
        self.assertIn("Executive Platform Analytics", res_analytics.get_data(as_text=True))

        # Test Admin ML Analytics
        res_ml = self.client.get('/admin/ml-analytics')
        self.assertEqual(res_ml.status_code, 200)
        self.assertIn("Machine Learning Model Comparison", res_ml.get_data(as_text=True))

    def test_07_admin_dedicated_subpages(self):
        """Test admin can access all dedicated admin subpages (Bookings, Cancellations, Revenue, Reports, Settings)."""
        self.client.post('/login', data={'login_mode': 'admin', 'username': 'admin', 'password': 'air123'})

        pages = {
            '/admin/bookings': 'Master Booking Management',
            '/admin/cancellations': 'Ticket Cancellations & Simulated Refunds',
            '/admin/revenue': 'Financial Turnover & Revenue Ledger',
            '/admin/reports': 'Platform Executive Reports',
            '/admin/settings': 'System Policy & Configuration Settings'
        }
        for route, title in pages.items():
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Expected 200 on {route}")
            self.assertIn(title, res.get_data(as_text=True), f"Expected '{title}' content in {route}")

if __name__ == '__main__':
    unittest.main()

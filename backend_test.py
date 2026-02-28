import requests
import sys
import json
import time
import os
from datetime import datetime
from pathlib import Path

class SquashSenseAPITester:
    def __init__(self, base_url="https://shot-lens.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.match_id = None

    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, is_json=True):
        """Run a single API test"""
        url = f"{self.api_base}{endpoint}"
        headers = {}
        if is_json and not files:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        self.log(f"Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                self.log(f"✅ PASSED - Status: {response.status_code}")
                
                # Try to parse JSON response
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        response_data = response.json()
                        return success, response_data
                    else:
                        return success, response.content
                except:
                    return success, response.text
            else:
                self.log(f"❌ FAILED - Expected {expected_status}, got {response.status_code}")
                self.log(f"   Response: {response.text[:200]}")

            return success, {}

        except Exception as e:
            self.log(f"❌ FAILED - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test basic health check"""
        return self.run_test(
            "Health Check",
            "GET",
            "/health",
            200
        )

    def test_get_empty_matches(self):
        """Test getting matches when none exist"""
        return self.run_test(
            "Get Matches (Empty)",
            "GET",
            "/matches",
            200
        )

    def test_upload_match(self):
        """Test uploading a video match"""
        # Create a dummy video file for testing
        dummy_video_content = b"fake video content for testing"
        files = {'file': ('test_match.mp4', dummy_video_content, 'video/mp4')}
        data = {'title': 'Test Match Upload'}
        
        success, response = self.run_test(
            "Upload Match",
            "POST",
            "/matches/upload",
            200,
            data=data,
            files=files,
            is_json=False
        )
        
        if success and 'id' in response:
            self.match_id = response['id']
            self.log(f"   Match ID: {self.match_id}")
        
        return success, response

    def test_get_matches_with_data(self):
        """Test getting matches after upload"""
        return self.run_test(
            "Get Matches (With Data)",
            "GET",
            "/matches",
            200
        )

    def test_get_specific_match(self):
        """Test getting a specific match by ID"""
        if not self.match_id:
            self.log("❌ SKIPPED - No match ID available")
            return False, {}
        
        return self.run_test(
            "Get Specific Match",
            "GET",
            f"/matches/{self.match_id}",
            200
        )

    def test_export_json(self):
        """Test JSON export"""
        if not self.match_id:
            self.log("❌ SKIPPED - No match ID available")
            return False, {}
        
        return self.run_test(
            "Export Match JSON",
            "GET",
            f"/matches/{self.match_id}/export/json",
            200,
            is_json=False
        )

    def test_export_pdf(self):
        """Test PDF export"""
        if not self.match_id:
            self.log("❌ SKIPPED - No match ID available")
            return False, {}
        
        return self.run_test(
            "Export Match PDF",
            "GET",
            f"/matches/{self.match_id}/export/pdf",
            200,
            is_json=False
        )

    def test_delete_match(self):
        """Test deleting a match"""
        if not self.match_id:
            self.log("❌ SKIPPED - No match ID available")
            return False, {}
        
        return self.run_test(
            "Delete Match",
            "DELETE",
            f"/matches/{self.match_id}",
            200
        )

    def test_get_nonexistent_match(self):
        """Test getting a match that doesn't exist"""
        return self.run_test(
            "Get Non-existent Match",
            "GET",
            "/matches/fake-id-12345",
            404
        )

    def test_invalid_file_upload(self):
        """Test uploading invalid file type"""
        files = {'file': ('test.txt', b"not a video file", 'text/plain')}
        data = {'title': 'Invalid File Test'}
        
        return self.run_test(
            "Upload Invalid File Type",
            "POST",
            "/matches/upload",
            400,
            data=data,
            files=files,
            is_json=False
        )

    def wait_for_analysis(self):
        """Wait for analysis to complete (up to 30 seconds)"""
        if not self.match_id:
            return False
            
        self.log("Waiting for analysis to complete...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{self.api_base}/matches/{self.match_id}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')
                    progress = data.get('progress', 0)
                    self.log(f"   Analysis status: {status} ({progress}%)")
                    
                    if status in ['completed', 'failed']:
                        return status == 'completed'
                        
            except Exception as e:
                self.log(f"   Error checking status: {e}")
            
            time.sleep(1)
        
        self.log("   Analysis timeout - continuing with tests")
        return False

def main():
    """Run all backend API tests"""
    tester = SquashSenseAPITester()
    
    print("=" * 60)
    print("SQUASHSENSE AI BACKEND API TESTING")
    print("=" * 60)
    print(f"Base URL: {tester.base_url}")
    print(f"API Base: {tester.api_base}")
    print("=" * 60)

    # Test sequence
    tests = [
        tester.test_health_check,
        tester.test_get_empty_matches,
        tester.test_upload_match,
        tester.test_get_matches_with_data,
        tester.test_get_specific_match,
        tester.test_export_json,
        tester.test_export_pdf,
        tester.test_get_nonexistent_match,
        tester.test_invalid_file_upload,
        tester.test_delete_match
    ]

    # Run tests
    for test_func in tests:
        try:
            test_func()
            print()  # Add spacing between tests
        except Exception as e:
            tester.log(f"❌ CRITICAL ERROR in {test_func.__name__}: {str(e)}")
            print()

    # Check analysis completion
    if tester.match_id:
        analysis_complete = tester.wait_for_analysis()
        if analysis_complete:
            tester.log("✅ Analysis completed successfully")
        else:
            tester.log("⚠️  Analysis did not complete in time")

    # Print summary
    print("=" * 60)
    print("BACKEND API TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    success_rate = (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print("=" * 60)

    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
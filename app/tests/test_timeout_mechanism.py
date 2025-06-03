"""
Test script to verify timeout mechanism works on both platforms
Run this to ensure Phase 1 foundation is solid
"""
import cv2 as cv
import time
import platform
import threading
import sys

def test_native_timeout_support():
    """Test if OpenCV supports native timeout"""
    print(f"Platform: {platform.system()}")
    print(f"OpenCV version: {cv.__version__}")
    
    has_timeout = hasattr(cv, 'CAP_PROP_OPEN_TIMEOUT_MSEC')
    print(f"Native timeout support: {has_timeout}")
    
    if has_timeout:
        # Test setting timeout
        try:
            cap = cv.VideoCapture()
            cap.set(cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
            print("✓ Can set CAP_PROP_OPEN_TIMEOUT_MSEC")
        except Exception as e:
            print(f"✗ Error setting timeout: {e}")
    
    return has_timeout

def test_thread_timeout(url, timeout_ms):
    """Test fallback thread-based timeout"""
    print(f"\nTesting thread-based timeout with {timeout_ms}ms...")
    
    result = [False, None]
    exception = [None]
    
    def read_thread():
        try:
            cap = cv.VideoCapture(url)
            result[0] = cap.isOpened()
            if result[0]:
                ret, frame = cap.read()
                result[1] = frame
            cap.release()
        except Exception as e:
            exception[0] = e
    
    start_time = time.time()
    thread = threading.Thread(target=read_thread)
    thread.daemon = True
    thread.start()
    
    # Wait for thread with timeout
    thread.join(timeout_ms / 1000.0)
    
    elapsed = (time.time() - start_time) * 1000
    
    if thread.is_alive():
        print(f"✓ Timeout triggered after {elapsed:.0f}ms")
        return False, None
    else:
        print(f"✓ Completed in {elapsed:.0f}ms")
        return result[0], result[1]

def test_blocking_read_simulation():
    """Simulate blocking read scenario"""
    print("\nSimulating blocking read scenario...")
    
    # Test with invalid URL that will timeout
    invalid_url = "rtsp://192.168.255.255:554/invalid"
    
    print("Testing with invalid URL (should timeout):")
    success, frame = test_thread_timeout(invalid_url, 2000)
    print(f"Result: {'Success' if success else 'Failed/Timeout'}")

def main():
    print("=" * 50)
    print("StreamWorker Timeout Mechanism Test")
    print("=" * 50)
    
    # Test 1: Check native support
    has_native = test_native_timeout_support()
    
    # Test 2: Test thread-based timeout
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        # Use webcam as default test
        test_url = 0
    
    print(f"\nTesting with source: {test_url}")
    
    # Test successful connection
    if test_url == 0:
        success, frame = test_thread_timeout(test_url, 3000)
        if success:
            print("✓ Successfully connected to webcam")
        else:
            print("✗ Failed to connect to webcam")
    
    # Test 3: Blocking scenario
    test_blocking_read_simulation()
    
    print("\n" + "=" * 50)
    print("Test complete!")
    
    # Summary
    print("\nRecommendations:")
    if has_native:
        print("• Use native OpenCV timeout (CAP_PROP_OPEN_TIMEOUT_MSEC)")
    else:
        print("• Use thread-based fallback timeout mechanism")
    print("• Set read timeout to 100-150ms for responsive operation")
    print("• Implement reconnection logic with cooldown period")

if __name__ == "__main__":
    main()
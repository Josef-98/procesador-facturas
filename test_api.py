#!/usr/bin/env python3
"""
Test script for the invoice processing API
"""
import requests
import json

def test_api():
    """Test the invoice processing endpoint"""
    
    # API endpoint
    url = "http://localhost:5000/procesar-factura"
    
    # Test with the sample invoice image
    image_path = "facturaPrueba.jpg"
    
    try:
        # Test home endpoint first
        print("🔍 Testing home endpoint...")
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            print("✅ Home endpoint working")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Home endpoint failed: {response.status_code}")
        
        print("\n" + "="*50 + "\n")
        
        # Test invoice processing
        print("📄 Testing invoice processing...")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Invoice processing successful!")
            print("\n📊 Processing Results:")
            print(f"Status: {result['status']}")
            print(f"Models available: {result['models_available']}")
            print(f"Processing mode: {result['data'].get('processing_mode', 'unknown')}")
            
            if 'data' in result:
                data = result['data']
                if 'detected_classes' in data:
                    print(f"Detected classes: {len(data['detected_classes'])}")
                    for class_name in data['detected_classes'][:5]:  # Show first 5
                        print(f"  - {class_name}")
                    if len(data['detected_classes']) > 5:
                        print(f"  ... and {len(data['detected_classes']) - 5} more")
            
            print(f"\n📋 Full response saved to 'test_result.json'")
            with open('test_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        else:
            print(f"❌ Invoice processing failed: {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error.get('message', 'Unknown error')}")
            except:
                print(f"Response: {response.text}")
                
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        print("Make sure facturaPrueba.jpg is in the current directory")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        print("Make sure the Flask app is running on http://localhost:5000")
        print("Run: python app.py")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Invoice Processing API")
    print("="*50)
    test_api()
from PriceTagApp import PriceTagApp


def main():
    print("=" * 50)
    print("🏷️  Price tag recognition app")
    print("=" * 50)
    print("\n⏳ Starting the app...")
    app = PriceTagApp()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        api_port=8000
    )

if __name__ == "__main__":
    main()
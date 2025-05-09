import random

def get_user_guess():
    """Prompt the user for a guess and handle invalid input."""
    while True:
        try:
            return int(input("Enter your guess: "))
        except ValueError:
            print("❌ Invalid input. Please enter a valid integer.")

def play_game():
    """Main game loop: the player tries to guess the randomly chosen number."""
    target = random.randint(1, 100)
    max_attempts = 10
    attempts = 0

    print("🎉 Welcome to the Number Guessing Game!")
    print(f"You have {max_attempts} attempts to guess the number between 1 and 100.")

    while attempts < max_attempts:
        guess = get_user_guess()
        attempts += 1

        if guess < target:
            print("⬇️ Too low!")
        elif guess > target:
            print("⬆️ Too high!")
        else:
            print(f"✅ Correct! You guessed the number in {attempts} attempts.")
            break
    else:
        print(f"❗ Out of attempts. The number was {target}.")

if __name__ == "__main__":
    play_game()

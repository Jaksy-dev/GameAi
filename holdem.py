from treys import Card, Evaluator, Deck
import random
import concurrent.futures

deck = Deck()
evaluator = Evaluator()
iterations=100000

my_hand = []
board = []
card_1 = input("Starting hand card 1 (rank, suit):")
card_2 = input("Starting hand card 2 (rank, suit):")
#todo check for invalid input
my_hand.append(Card.new(card_1))
my_hand.append(Card.new(card_2))

deck.cards.remove(my_hand[0])
deck.cards.remove(my_hand[1])

scores = {
    "win" : 0,
    "draw":0,
    "lose" : 0
}

# Pre-flop

for _ in range(iterations):

    hand_2 = deck.draw(2)
    board_pre_flop = deck.draw(5)

    my_hand_eval = evaluator.evaluate(board_pre_flop, my_hand)
    hand_2_eval = evaluator.evaluate(board_pre_flop, hand_2)

    if my_hand_eval < hand_2_eval:
        scores["win"]+=1
    elif my_hand_eval == hand_2_eval:
        scores["draw"]+=1
    else:
        scores["lose"]+=1

    # Reshuffle drawn cards
    deck.cards = deck.cards + hand_2 + board_pre_flop
    random.shuffle(deck.cards)

win_chance = float(scores["win"]) / float(iterations)
draw_chance = float(scores["draw"]) / float(iterations)
lose_chance = float(scores["lose"]) / float(iterations)

print("Pre-flop")
Card.print_pretty_cards(my_hand)
print(f"Win chance: {win_chance * 100:.2f}%, Draw chance: {draw_chance * 100:.2f}%, Lose chance: {lose_chance * 100:.2f}%")

# Flop

flop_1 = input("Flop card 1 (rank, suit):")
flop_2 = input("Flop card 2 (rank, suit):")
flop_3 = input("Flop card 3 (rank, suit):")

board.append(Card.new(flop_1))
board.append(Card.new(flop_2))
board.append(Card.new(flop_3))

deck.cards.remove(board[0])
deck.cards.remove(board[1])
deck.cards.remove(board[2])

scores = {
    "win" : 0,
    "draw":0,
    "lose" : 0
}

for n in range(iterations):

    hand_2 = deck.draw(2)
    
    board_flop = board + deck.draw(2)
    
    my_hand_eval = evaluator.evaluate(board_flop, my_hand)
    hand_2_eval = evaluator.evaluate(board_flop, hand_2)

    if my_hand_eval < hand_2_eval:
        scores["win"]+=1
    elif my_hand_eval == hand_2_eval:
        scores["draw"]+=1
    else:
        scores["lose"]+=1

    # Reshuffle drawn cards
    deck.cards = deck.cards + hand_2 + board_flop[3:5]
    random.shuffle(deck.cards)

win_chance = float(scores["win"]) / float(iterations)
draw_chance = float(scores["draw"]) / float(iterations)
lose_chance = float(scores["lose"]) / float(iterations)

print("Flop")
Card.print_pretty_cards(board)
Card.print_pretty_cards(my_hand)
print(evaluator.class_to_string(evaluator.get_rank_class(evaluator.evaluate(board, my_hand))))
print(f"Win chance: {win_chance * 100:.2f}%, Draw chance: {draw_chance * 100:.2f}%, Lose chance: {lose_chance * 100:.2f}%")

# Turn

turn = input("Turn card (rank, suit):")

board.append(Card.new(turn))

deck.cards.remove(board[3])

scores = {
    "win" : 0,
    "draw":0,
    "lose" : 0
}

for n in range(iterations):

    hand_2 = deck.draw(2)
    
    board_turn = board + deck.draw(1)

    my_hand_eval = evaluator.evaluate(board_turn, my_hand)
    hand_2_eval = evaluator.evaluate(board_turn, hand_2)

    if my_hand_eval < hand_2_eval:
        scores["win"]+=1
    elif my_hand_eval == hand_2_eval:
        scores["draw"]+=1
    else:
        scores["lose"]+=1

    # Reshuffle drawn cards
    deck.cards = deck.cards + hand_2 + [board_turn[4]]
    random.shuffle(deck.cards)

win_chance = float(scores["win"]) / float(iterations)
draw_chance = float(scores["draw"]) / float(iterations)
lose_chance = float(scores["lose"]) / float(iterations)

print("Turn")
Card.print_pretty_cards(board)
Card.print_pretty_cards(my_hand)
print(evaluator.class_to_string(evaluator.get_rank_class(evaluator.evaluate(board, my_hand))))
print(f"Win chance: {win_chance * 100:.2f}%, Draw chance: {draw_chance * 100:.2f}%, Lose chance: {lose_chance * 100:.2f}%")

# River

river = input("River card (rank, suit):")

board.append(Card.new(river))

deck.cards.remove(board[4])

scores = {
    "win" : 0,
    "draw":0,
    "lose" : 0
}

for n in range(iterations):

    hand_2 = deck.draw(2)
    
    my_hand_eval = evaluator.evaluate(board, my_hand)
    hand_2_eval = evaluator.evaluate(board, hand_2)

    if my_hand_eval < hand_2_eval:
        scores["win"]+=1
    elif my_hand_eval == hand_2_eval:
        scores["draw"]+=1
    else:
        scores["lose"]+=1

    # Reshuffle drawn cards
    deck.cards = deck.cards + hand_2
    random.shuffle(deck.cards)

win_chance = float(scores["win"]) / float(iterations)
draw_chance = float(scores["draw"]) / float(iterations)
lose_chance = float(scores["lose"]) / float(iterations)

print("River")
Card.print_pretty_cards(board)
Card.print_pretty_cards(my_hand)
print(evaluator.class_to_string(evaluator.get_rank_class(evaluator.evaluate(board, my_hand))))
print(f"Win chance: {win_chance * 100:.2f}%, Draw chance: {draw_chance * 100:.2f}%, Lose chance: {lose_chance * 100:.2f}%")

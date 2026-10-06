from sentence_transformers import SentenceTransformer, util
import numpy as np

movies = [
    {
        "title": "Project Hail Mary",
        "plot": "A lone surviving astronaut awakens with amnesia on a desperate deep-space voyage to save Earth from an extinction-level solar crisis.\nTeaming up with an unlikely alien scientist, he uses intellect and ingenuity to find a solution before time runs out."
    },
    {
        "title": "The Odyssey",
        "plot": "King Odysseus embarks on a perilous decade-long voyage across the Mediterranean following the fall of Troy.\nHe must outwit mythical monsters, vengeful deities, and deadly temptations to reclaim his throne and family."
    },
    {
        "title": "Obsession",
        "plot": "A passionate, forbidden affair between two people spirals into intense psychological fixation and deceit.\nAs hidden secrets come to light, their boundary between deep desire and self-destruction dangerously blurs."
    },
    {
        "title": "One Battle After Another",
        "plot": "A former revolutionary is pulled back into a chaotic underground world to protect those he loves.\nCaught between corrupt authorities and relentless rivals, he is forced to fight through a web of shifting allegiances."
    },
    {
        "title": "Spider-Man: Brand New Day",
        "plot": "Peter Parker navigates life stripped of his high-tech connections and erased from the memories of everyone he loves.\nForced back to street-level roots, he battles formidable criminal syndicates to protect New York City on his own terms."
    },
    {
        "title": "Frankenstein",
        "plot": "A brilliant yet hubristic scientist successfully breathes life into an artificial creature assembled from human remains.\nHorrified by his creation, he abandons it, igniting a tragic and monstrous cycle of vengeance and loneliness."
    },
    {
        "title": "Interstellar",
        "plot": "A team of explorers travels through an enigmatic wormhole near Saturn to find a habitable new home for a dying humanity.\nA pilot must risk never seeing his children again as extreme gravitational time dilation threatens their mission."
    },
    {
        "title": "Wake Up Dead Man",
        "plot": "Famed private investigator Benoit Blanc arrives to unravel a complex, high-stakes murder mystery with eccentric suspects.\nHe methodically sifts through layered lies, dark family secrets, and deceptive motives to uncover the killer."
    },
    {
        "title": "Sinners",
        "plot": "Twin brothers return to their rural Southern hometown hoping to leave their turbulent pasts behind.\nTheir homecoming turns into a fight for survival when an ancient, sinister evil emerges to torment the community."
    },
    {
        "title": "Weapons",
        "plot": "An interrelated mystery follows the sudden and inexplicable disappearance of high school students in a quiet town.\nThe tragedy unravels the social fabric of the community, revealing chilling forces lurking beneath suburban tranquility."
    },
    {
        "title": "Bugonia",
        "plot": "Two conspiracy-obsessed kidnappers abduct a high-powered pharmaceutical CEO, convinced she is an alien plotting Earth's doom.\nTheir tense interrogation rapidly devolves into a dark, chaotic psychological battle of wits."
    },
    {
        "title": "Marty Supreme",
        "plot": "A charismatic table tennis prodigy navigates the fast-moving subcultures and gambling rings of 1950s New York.\nDriven by an unyielding desire for glory, he pushes his personal life to the brink to become an undisputed champion."
    },
    {
        "title": "The Dark Knight",
        "plot": "Batman, Lieutenant Jim Gordon, and DA Harvey Dent form an alliance to rid Gotham City of organized crime.\nTheir efforts are violently shattered by the Joker, a psychotic mastermind intent on plunging the city into moral anarchy."
    },
    {
        "title": "Avatar: Fire and Ash",
        "plot": "Jake Sully and Neytiri encounter a hostile, volcanic clan of Na'vi known as the Ash People on Pandora.\nThe family must navigate intense new tribal warfare alongside the continuing threat from human corporate invaders."
    },
    {
        "title": "Michael",
        "plot": "A biographical drama detailing the life, extraordinary musical talent, and private struggles of Michael Jackson.\nIt tracks his rise from a child star in Motown to his reign as a global cultural icon."
    },
    {
        "title": "Fight Club",
        "plot": "An insomniac white-collar worker forms a subterranean bare-knuckle fighting club with a charismatic soap salesman.\nThe underground circle rapidly evolves into an anti-consumerist insurgent movement with shocking revelations."
    },
    {
        "title": "The Shawshank Redemption",
        "plot": "A quiet banker is wrongfully sentenced to two consecutive life terms inside the brutal Shawshank State Penitentiary.\nOver several decades, he forms a deep bond with fellow inmate Red and secretly plots an ingenious bid for freedom."
    },
    {
        "title": "F1: The Movie",
        "plot": "A veteran Formula 1 driver comes out of retirement to mentor a promising young rookie at a struggling underdog team.\nTogether, they push their speed and psychological limits against the fiercest competitors in global motorsport."
    },
    {
        "title": "Inception",
        "plot": "A master thief who infiltrates dreams to steal corporate secrets is offered a chance to wipe his criminal history clean.\nTo succeed, he must assemble a specialized team to perform the near-impossible task of planting an idea inside a target's mind."
    },
    {
        "title": "Backrooms",
        "plot": "A videographer accidentally slips through the seams of reality into an infinite, yellow-carpeted labyrinth of empty rooms.\nHe must navigate disorienting liminal spaces and lurking supernatural entities to find a way back home."
    }
]
def main():
    while True:
        query = input("Enter your query: ")
        if all(char.isalpha() or char.isspace() for char in query):
            break
        else:
            print("Invalid input. Please enter a query containing only letters and spaces.")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    movies_embedding = model.encode([movie['plot'] for movie in movies], convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)


    similarity = util.cos_sim(query_embedding, movies_embedding)[0]
    similarity_np = similarity.cpu().numpy()

    top_indices = np.argsort(similarity_np)[::-1][:3]


    print("/the top 3 movies that match your query are:\n")
    for rank, index in enumerate(top_indices, start = 1):
        print(f"{rank}. {movies[index]['title']} - Similarity Score: {similarity_np[index]:.4f}")
        print(f"   Plot: {movies[index]['plot']}\n")
    print(f"these are the movies for the query: {query}")

if __name__ == "__main__":
    main()
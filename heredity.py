import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    # Initialize the probability value to 1 
    probability = float(1)

    for person in people:
        # Determine the number of genes the person has
        num_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        
        # Determine if the person has the trait or not
        has_trait_person = person in have_trait
        
        # Get the person's mother and father
        mother = people[person]["mother"]
        father = people[person]["father"]

        # If no parents, use the gene in the probability
        if mother is None and father is None:
            probability *= PROBS["gene"][num_genes]

        else:
            # Calculate the passing gene probability from the parents
            prob_passing_mother = 1 - PROBS["mutation"] if mother in two_genes else 0.5 if mother in one_gene else PROBS["mutation"]
            prob_passing_father = 1 - PROBS["mutation"] if father in two_genes else 0.5 if father in one_gene else PROBS["mutation"]

            # Calculate the probability of the person's gene configuration
            if num_genes == 2:
                probability *= prob_passing_mother * prob_passing_father
            elif num_genes == 1:
                probability *= prob_passing_mother * (1 - prob_passing_father) + (1 - prob_passing_mother) * prob_passing_father
            else:
                probability *= (1 - prob_passing_mother) * (1 - prob_passing_father)

        # Multiply the probability by the prob of having the trait
        probability *= PROBS["trait"][num_genes][has_trait_person]

    # Return the calculated joint probability
    return probability

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    # Iterate through each person in probabilities
    for person in probabilities:
        # Determine the number of genes the person has
        num_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        
        # Determine if the person has the trait or not
        has_trait_person = person in have_trait
        
        # Update gene distribution for the person
        probabilities[person]["gene"][num_genes] += p
        
        # Update trait distribution for the person
        probabilities[person]["trait"][has_trait_person] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).

    * Total gene sum: 0.4 + 0.3 + 0.3 = 1.0
    * Normalize each gene probability:
        0 genes: 0.4 / 1.0 = 0.4
        1 gene: 0.3 / 1.0 = 0.3
        2 genes: 0.3 / 1.0 = 0.3
    """

    # Loop through each person in probabilities
    for person in probabilities:
        # Normalize "gene" distribution
        gene_dist = probabilities[person]["gene"]
        total_gene_prob = sum(gene_dist.values())
        for num_genes in gene_dist:
            gene_dist[num_genes] /= total_gene_prob
        
        # Normalize "trait" distribution
        trait_dist = probabilities[person]["trait"]
        total_trait_prob = sum(trait_dist.values())
        for has_trait in trait_dist:
            trait_dist[has_trait] /= total_trait_prob


if __name__ == "__main__":
    main()

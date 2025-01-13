from .suprise import * 
from .boundaries_refinement import * 

def filter_boundaries(boundaries):
  l = 0
  boundaries_distances = []
  for b in boundaries:
    boundaries_distances.append(b-l)
    l = b

  mean_boundary = sum(boundaries_distances) / len(boundaries_distances)
  mean_error = sum([(i-mean_boundary) **2 **0.5 for i in boundaries_distances]) / len(boundaries_distances)

  return (mean_error/mean_boundary)


def episodic_suprise_setup_v1(text,
                           model,
                           processor,
                           config):
  tokens, surprise_scores = compute_surprise_scores(text, model, processor, batch_size=config.batch_size)
  print("Surprise scores:", str(surprise_scores)[:100])

  similarity_matrix = similarity_fn(text, processor.tokenizer, model)

  return surprise_scores, similarity_matrix, tokens

def episodic_suprise_setup_v2(
    surprise_scores,
    similarity_matrix,
    tokens,
    config ):

  mean_surprise = np.mean(surprise_scores)
  variance_surprise = np.var(surprise_scores)
  scaling_factor = config.scaling_factor

  # Calculate the threshold
  if config.threshold is None:
    threshold = mean_surprise + scaling_factor * variance_surprise
  else:
    threshold = config.threshold

  initial_boundaries = get_initial_boundaries(surprise_scores, threshold)
  refined_boundaries = boundary_refinement(similarity_matrix, initial_boundaries)
  # take only boundaries that more than 2 event

  print("Initial Boundaries:", initial_boundaries)
  print("Refined Boundaries:", refined_boundaries)
  # Split into initial events
  events = []
  start = 0

  for boundary in refined_boundaries[1:]:
      events.append(''.join(tokens[start:boundary+1]))
      start = boundary+1
  events.append(tokens[start:])

  print("Refined Events:")
  for i, event in enumerate(events):
      print(f"Event {i+1}: {(event)}")

  return events, refined_boundaries
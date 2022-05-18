from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F

class SymbolicEncoder(nn.Module):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.embed_size = embedding_size
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)
    self.modules = [self.fc1, self.fc2, self.fc3]

  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(nn.Module):
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()
    self.embed_size = embedding_size
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
    self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    return hidden


class VisualEncoderSmall(nn.Module):
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()
    self.embed_size = embedding_size
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
    self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding='same')
    self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding='same')
    self.conv4 = nn.Conv2d(64, 128, 5, stride=1)
    self.fc = nn.Identity() if embedding_size == 128 else nn.Linear(128, embedding_size)
    self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 128)
    hidden = self.fc(hidden)  # Identity if embedding size is 128 else linear projection
    return hidden

def Encoder(observation_size: Tuple[int], embedding_size: int, activation_function='relu'):
  if observation_size == (3, 64, 64):
    return VisualEncoder(embedding_size, activation_function)
  elif len(observation_size) == 3:
    return VisualEncoderSmall(embedding_size, activation_function)
  elif len(observation_size) == 1:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    raise NotImplementedError

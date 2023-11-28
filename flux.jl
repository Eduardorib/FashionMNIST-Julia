using Flux, MLDatasets, Statistics, ProgressMeter, MLUtils, Dates, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf, BSON

include("TermShow.jl")

# Training data
train_data = FashionMNIST(split=:train)
train_X, train_Y = train_data[:]
categories = train_data.metadata["class_names"]

train_X = MLUtils.unsqueeze(train_X, 3)
train_Y = Flux.onehotbatch(train_Y, 0:(length(categories)-1))

# Test Data
test_data = FashionMNIST(split=:test)
test_X, test_Y = test_data[:]
test_X = MLUtils.unsqueeze(test_X, 3)

# Model
model = Chain(
  Conv((5, 5), 1 => 6, relu),
  MaxPool((2, 2)),
  Conv((5, 5), 6 => 16, relu),
  MaxPool((2, 2)),
  Flux.flatten,
  Dense(prod((4, 4, 16)), 120, relu),
  Dense(120, 84, relu),
  Dense(84, length(categories))
)

accuracy(ŷ, y) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))

# Loss Function
loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

# Optmiser
optimiser = ADAM()

# Training
function trainModel()
  epochs = 40
  melhor_acu = 0.0

  println("Training with ", epochs, " epochs...")
  parameters = Flux.params(model)
  _train_data = [(train_X, train_Y)]

  @showprogress for epoch in 1:epochs
    Flux.train!(loss, parameters, _train_data, optimiser)

    test_ŷ = model(test_X)
    acu = accuracy()

    if acu >= melhor_acu
      @info(" -> Uma nova melhor acurácia! Salvando o modelo para mnist_conv.bson")
      BSON.@save joinpath("./", "mnist_conv.bson") params = parameters epoch acu
      melhor_acu = acu
    end

    println("\n")
    @info(@sprintf("\n[%d]: Acurácia nos testes: %.4f", epoch, acu))
    # Se a acurácia for muito boa, termine o treino
    if acu >= 0.999
      @info(" -> Training done : accuracy of 99.9%")
      break
    end
  end
  println("Training done with ", epochs, " epochs. Max accuracy: ", melhor_acu)
end

function accuracy()
  a_sum_ = 0
  for test in eachindex(test_Y)
    local temp_image_ = test_X[:, :, 1, test]
    local temp_image_r = reshape(temp_image_, size(temp_image_)..., 1, 1)
    guess_ = findmax(model(temp_image_r))[2][1]
    correct = test_Y[test] + 1
    if guess_ == correct
      a_sum_ = a_sum_ + 1
    end
  end

  return a_sum_ / length(test_Y)
end

function testModel()
  println("Testing model...")

  # Run inference on the first test image which should be "Ankle Boot"
  temp_image_ = test_X[:, :, 1, 1]
  temp_image_r = reshape(temp_image_, size(temp_image_)..., 1, 1)

  TermShow.hires_render_greyscale_image(temp_image_r)

  guess = findmax(model(temp_image_r))[2]
  println("This should be an 'Ankle boot': ", categories[guess])
end


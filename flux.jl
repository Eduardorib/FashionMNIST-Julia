using Flux, MLDatasets, Statistics, ProgressMeter, MLUtils, Dates, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf, BSON

include("termShow.jl")

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
test_ŷ = Nothing
# test_Y = Flux.onehotbatch(test_Y, 0:(length(categories)-1))

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

# Loss Function
loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

# Optmiser
optimiser = ADAM()

parameters = Flux.params(model);

BSON.@load joinpath("./", "mnist_conv.bson") params

# Loading params
Flux.loadparams!(model, params)

# Training
function trainModel()
  epochs = 100000
  melhor_acu = 0.0

  println("Training with ", epochs, " epochs...")
  _train_data = [(train_X, train_Y)]

  @showprogress for epoch in 1:epochs
    Flux.train!(loss, parameters, _train_data, optimiser)

    global test_ŷ = model(test_X)
    acu = accuracy()

    println("\n")
    @info(@sprintf("\n[%d]: Accuracy on test: %.4f", epoch, acu))

    if acu >= melhor_acu
      println("\n")
      @info(" -> Better accuracy found! Salving model to mnist_conv.bson")
      BSON.@save joinpath("./", "mnist_conv.bson") params = parameters epoch acu
      melhor_acu = acu
    end

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

  test = rand(1:length(test_Y))

  local temp_image_ = test_X[:, :, 1, test]
  local temp_image_r = reshape(temp_image_, size(temp_image_)..., 1, 1)

  TermShow.hires_render_greyscale_image(temp_image_r)

  guess_ = findmax(model(temp_image_r))[2][1]
  correct = test_Y[test] + 1
  println("Guessed category: ", categories[guess_], "\nCorrect category: ", categories[correct])
  if guess_ == correct
    println("\nCorrect guess!")
  end
end

trainModel()

y_teste = Flux.onehotbatch(test_Y, 0:(length(categories)-1))

a, b = Flux.onecold(y_teste) .- 1, Flux.onecold(test_ŷ) .- 1
println(typeof(a), " ", typeof(b))
println(length(a), " ", length(b))

cm = ConfusionMatrix()
fit!(cm, a, b)
print(cm)

res = info(cm)

heatmap(string.(res["categories"]),
  string.(res["categories"]),
  res["normalised_scores"],
  seriescolor=cgrad([:white, :blue]),
  xlabel="Predito",
  ylabel="Real",
  title="Matriz de Confusão (scores normalizados)")

# Limita o mapa de cores, para vermos melhor onde os erros estão
import { quizData } from "./data.js";

const quizContainer = document.getElementById("quiz-container");
const submitButton = document.getElementById("submit-btn");
const resultContainer = document.getElementById("result-container");
const attemptsCount = document.getElementById("attempts-count");

const totalQuestions = 23;
let currentQuestionIndex = 0;
let score = 0;
let attempts = 0;

// Shuffle function to randomize array elements
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function displayQuestion() {
  if (currentQuestionIndex >= totalQuestions) {
    quizContainer.innerHTML = "";
    quizContainer.textContent = "You have attempted all questions.";
    submitButton.style.display = "none";
    return;
  }

  quizContainer.innerHTML = "";
  const currentQuestion = quizData[currentQuestionIndex];
  const shuffledOptions = [...currentQuestion.options];
  shuffleArray(shuffledOptions);

  const questionDiv = document.createElement("div");
  questionDiv.classList.add(
    "border",
    "border-gray-300",
    "rounded-lg",
    "p-6",
    "mb-8"
  );

  const questionText = document.createElement("p");
  questionText.classList.add("text-lg", "font-semibold", "mb-4");
  questionText.innerHTML = splitAndWrapCode(currentQuestion.question); // Modified to call splitAndWrapCode function
  questionDiv.appendChild(questionText);

  shuffledOptions.forEach((option, index) => {
    const optionLabel = document.createElement("label");
    optionLabel.classList.add("block", "cursor-pointer", "mb-4"); // Increased margin-bottom for spacing
    const optionInput = document.createElement("input");
    optionInput.type = "radio";
    optionInput.name = "question";
    optionInput.value = option;
    optionInput.id = `option-${index}`;
    optionInput.addEventListener("change", () => {
      attempts++;
      updateAttempts();
    });
    optionLabel.appendChild(optionInput);
    optionLabel.innerHTML += option;
    questionDiv.appendChild(optionLabel);
  });

  quizContainer.appendChild(questionDiv);
}

function splitAndWrapCode(text) {
  const parts = text.split("```");
  let result = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      result += parts[i];
    } else {
      result += `<code  >${parts[i]}</code>`;
    }
  }
  return result;
}

function calculateScore() {
  const selectedOption = document.querySelector(
    'input[name="question"]:checked'
  );
  if (selectedOption) {
    if (selectedOption.value === quizData[currentQuestionIndex].correctAnswer) {
      score++;
    }
  }
}

function displayResult() {
  resultContainer.innerHTML = "";
  const resultTitle = document.createElement("h2");
  resultTitle.classList.add("text-2xl", "font-semibold", "mb-4");
  resultTitle.textContent = `Your score: ${score}/${totalQuestions}`;
  resultContainer.appendChild(resultTitle);

  const answersList = document.createElement("ul");
  quizData.forEach((quiz, index) => {
    const answerItem = document.createElement("li");
    answerItem.classList.add("text-lg", "mb-2");
    answerItem.textContent = `${quiz.question} - Correct Answer: ${quiz.correctAnswer}`;
    answersList.appendChild(answerItem);
  });
  resultContainer.appendChild(answersList);
}

function updateAttempts() {
  attemptsCount.textContent = attempts;
}

submitButton.addEventListener("click", () => {
  calculateScore();
  currentQuestionIndex++;
  if (currentQuestionIndex < totalQuestions) {
    displayQuestion();
  } else {
    submitButton.style.display = "none";
    displayResult();
  }
});

shuffleArray(quizData);
displayQuestion();
updateAttempts();

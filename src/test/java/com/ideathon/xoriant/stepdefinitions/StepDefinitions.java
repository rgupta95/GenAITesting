package com.ideathon.xoriant.stepdefinitions;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static dev.langchain4j.data.message.UserMessage.userMessage;
//import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_3_5_TURBO;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_3_5_TURBO;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4;
import static java.time.Duration.ofSeconds;
import static java.util.stream.Collectors.joining;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.*;

import dev.langchain4j.model.openai.OpenAiChatModelName;
import dev.langchain4j.model.openai.OpenAiModelName.*;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.util.CoreMap;

import java.io.File;
import java.util.*;
import java.nio.file.Paths;

import com.ideathon.xoriant.Constants;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
//import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiModelName;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import io.cucumber.java.en.And;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import net.serenitybdd.core.Serenity;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StepDefinitions {

    private static final Logger logger = LoggerFactory.getLogger(StepDefinitions.class);
    EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    private static WordVectors wordVectors;
    private static StanfordCoreNLP pipeline;
    static {
        try {
            wordVectors = WordVectorSerializer.loadStaticModel(new File("src/test/resources/glove.6B.100d.txt"));
        } catch (Exception e) {
            logger.atError();
        }
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse");
        pipeline = new StanfordCoreNLP(props);
    }

    @Then("verify the accuracy of LLM response generated with response criteria having values like matching keywords {string} , minimum length {string}, punctuation {string} and expected response {string}")
    public static boolean verify_the_accuracy_of_LLM_response(String keywords,String minLength, String punctuation, String expectedResponse) {
        ResponseCriteria criteria = new ResponseCriteria(
                Arrays.asList(keywords.split("\\s+")),
              // Minimum length
                Integer.parseInt(minLength),
              // Ending punctuation
                Arrays.asList(punctuation.split("\\s+")),
                expectedResponse
        );
        AiMessage aiMessage;
        aiMessage = Serenity.sessionVariableCalled(Constants.AI_MESSAGE_RESPONSE);
        String response = aiMessage.text();
        if (response == null || response.isEmpty()) {
            return false;
        }
        // Normalize the response (e.g., convert to lowercase, remove punctuation)
        String normalizedResponse = normalizeText(response);

        // Convert response to lowercase for case-insensitive comparison
        String lowerCaseResponse = response.toLowerCase();
        // Count how many keywords are found in the response
        long matchedKeywordsCount = criteria.getRequiredKeywords().stream().filter(keyword -> {
            String lowerCaseKeyword = keyword.toLowerCase();
         //   boolean containsKeyword = lowerCaseResponse.contains(lowerCaseKeyword);
            boolean containsKeyword = normalizedResponse.contains(lowerCaseKeyword);
            System.out.println("Checking keyword '" + keyword + "' in response: " + containsKeyword);
            return containsKeyword;
        }).count();

        // Calculate the matching percentage
        double matchingPercentage = (double) matchedKeywordsCount / criteria.getRequiredKeywords().size() * 100;
        logger.info("Matching Percentage: " + matchingPercentage + "%");
        System.out.println("Matching Percentage: " + matchingPercentage + "%");
        // Set a threshold for the matching percentage (e.g., 70%)
        double threshold = 70.0;
        boolean meetsKeywordThreshold = matchingPercentage >= threshold;
        logger.info("Meets keyword threshold (" + threshold + "%): " + meetsKeywordThreshold);
        System.out.println("Meets keyword threshold (" + threshold + "%): " + meetsKeywordThreshold);

//
        // Check for minimum length
        boolean meetsMinLength = response.length() >= criteria.getMinLength();

        // Check for proper sentence endings
        boolean hasProperEndings = criteria.getEndingPunctuation().stream().anyMatch(response::endsWith);

        // Check semantic similarity
        boolean meetsSemanticSimilarity = checkSemanticSimilarity(response, criteria.getExpectedResponse()) > 0.75;

        // Check grammatical correctness
        boolean isGrammaticallyCorrect = checkGrammar(response);

        // Validate response based on criteria
        return meetsKeywordThreshold && meetsMinLength && hasProperEndings && meetsSemanticSimilarity && isGrammaticallyCorrect;
    }
    private static String normalizeText(String text) {
        // Example normalization: Convert to lowercase and remove punctuation
        return text.toLowerCase().replaceAll("[^a-z0-9 ]", "").trim();
    }

    private static double checkSemanticSimilarity(String response, String expectedResponse) {
        INDArray responseVector = getSentenceVector(response);
        INDArray expectedVector = getSentenceVector(expectedResponse);

        if (responseVector == null || expectedVector == null) {
            return 0.0;
        }

        return Transforms.cosineSim(responseVector, expectedVector);
    }

    private static INDArray getSentenceVector(String sentence) {
        String[] words = sentence.split("\\s+");
        INDArray vector = null;

        for (String word : words) {
            INDArray wordVector = wordVectors.getWordVectorMatrix(word);
            if (wordVector != null) {
                if (vector == null) {
                    vector = wordVector;
                } else {
                    vector.addi(wordVector);
                }
            }
        }

        if (vector != null) {
            vector.divi(words.length);
        }

        return vector;
    }

    private static boolean checkGrammar(String response) {
        Annotation document = new Annotation(response);
        pipeline.annotate(document);
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                if (pos.equals("UH")) {
                    return false;
                }
            }
        }
        return true;
    }

    public static class ResponseCriteria {
        private final List<String> requiredKeywords;
        private final int minLength;
        private final List<String> endingPunctuation;
        private final String expectedResponse;

        public ResponseCriteria(List<String> requiredKeywords, int minLength, List<String> endingPunctuation, String expectedResponse) {
            this.requiredKeywords = requiredKeywords;
            this.minLength = minLength;
            this.endingPunctuation = endingPunctuation;
            this.expectedResponse = expectedResponse;
        }

        public List<String> getRequiredKeywords() {
            return requiredKeywords;
        }

        public int getMinLength() {
            return minLength;
        }

        public List<String> getEndingPunctuation() {
            return endingPunctuation;
        }

        public String getExpectedResponse() {
            return expectedResponse;
        }
    }
    @Given("Generate prompt template with variable {word} about {word}")
    public void prompt_template_with_getVariable(String arg1, String arg2) {
        PromptTemplate promptTemplate = PromptTemplate.from("Tell me something  {{adjective}}  about {{content}}..");
        Map<String, Object> variables = new HashMap<>();
        variables.put("adjective", arg1);
        variables.put("content", arg2);
        Prompt prompt = promptTemplate.apply(variables);
        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(Constants.OPENAI_API_KEY)
                .modelName(GPT_3_5_TURBO)
                .temperature(0.3)
                .build();
        String response = model.generate(prompt.text());
        Serenity.setSessionVariable(Constants.PROMPT_RESPONSE).to(response);
    }

    @Then("verify the LLM response generated for response {string}")
    public void verify_llm_response(String response) {
        response = Serenity.sessionVariableCalled(Constants.PROMPT_RESPONSE);
        assertNotNull(response);
    }

    @Then("verify the LLM response generated from chat memory {word}")
    public void verify_chat_response(String response) {
        AiMessage aiMessage;
        aiMessage = Serenity.sessionVariableCalled(Constants.AI_MESSAGE_RESPONSE);
        response = aiMessage.text();
        assertNotNull(response);

    }

    @Then("verify the LLM response contains {string}")
    public void verify_chat_response_contains(String data) {
        AiMessage aiMessage;
        aiMessage = Serenity.sessionVariableCalled(Constants.AI_MESSAGE_RESPONSE);
        assertThat(aiMessage.text()).contains(data);
    }

    @Then("verify the LLM response contains {string} words")
    public void count_words_in_given_data(String data) {
        AiMessage aiMessage;
        aiMessage = Serenity.sessionVariableCalled(Constants.AI_MESSAGE_RESPONSE);
        assertEquals(Integer.parseInt(data), countWords(aiMessage.text()));

    }

    public int countWords(String data) {
        return (data == null || data.isEmpty()) ? 0 : data.trim().split("\\s+").length;

    }

    @Then("verify the LLM response with expected response {string}")
    public void verify_chat_response_contains_(String expectedOutput) {

        AiMessage aiMessage;
        aiMessage = Serenity.sessionVariableCalled(Constants.AI_MESSAGE_RESPONSE);

        assertThat(aiMessage.text()).contains(expectedOutput);
    }



    @Given("I created chat memory with LLM model")
    public void generate_chat_memory() {
        ChatMemory chatMemory = TokenWindowChatMemory.withMaxTokens(300, new OpenAiTokenizer(GPT_4));
        Serenity.setSessionVariable(Constants.CHAT_MEMORY_RESPONSE).to(chatMemory);
        logger.info("Chat memory is created"+ chatMemory);
    }

    @And("I add input data {string} to chat memory")
    public void addInput_toModel(String data) {
        ChatMemory chatMemory;
        ChatLanguageModel model = OpenAiChatModel.withApiKey(Constants.OPENAI_API_KEY);
        chatMemory = Serenity.sessionVariableCalled(Constants.CHAT_MEMORY_RESPONSE);
        chatMemory.add(userMessage(data));
        AiMessage answer = model.generate(chatMemory.messages())
                .content();
        Serenity.setSessionVariable(Constants.AI_MESSAGE_RESPONSE).to(answer);
    }

    @Given("I upload a document {string} and create word embeddings")
    public void create_word_embeddings_From_input_file(String documentName) {
        Document document = loadDocument(Paths.get("src/test/resources/" + documentName));
        DocumentSplitter splitter = DocumentSplitters.recursive(100, 0, new OpenAiTokenizer(OpenAiModelName.GPT_3_5_TURBO));
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments)
                .content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        Serenity.setSessionVariable(Constants.CREATE_EMBEDDING_STORE).to(embeddingStore);
    }

    @Then("I ask LLM question {string} based on stored word embeddings")
    public void i_ask_LLM_question(String question) {
        EmbeddingStore<TextSegment> embeddingStore;
        embeddingStore = Serenity.sessionVariableCalled(Constants.CREATE_EMBEDDING_STORE);
        Embedding questionEmbedding = embeddingModel.embed(question)
                .content();
        int maxResults = 5;
        double minScore = 1;
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.findRelevant(questionEmbedding, maxResults, minScore);
        PromptTemplate promptTemplate = PromptTemplate.from("Answer the following question to the best of your ability:\n" + "\n" + "Question:\n" + "{{question}}\n" + "\n" + "Base your answer on the following information:\n" + "{{information}}");
        String information = relevantEmbeddings.stream()
                .map(match -> match.embedded()
                        .text())
                .collect(joining("\n\n"));
        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("information", information);
        Prompt prompt = promptTemplate.apply(variables);
        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(Constants.OPENAI_API_KEY)
                .timeout(ofSeconds(60))
                .build();
        AiMessage aiMessage = chatModel.generate(prompt.toUserMessage())
                .content();
        Serenity.setSessionVariable(Constants.AI_MESSAGE_RESPONSE).to(aiMessage);
    }
}

package com.ideathon.xoriant.stepdefinitions;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static dev.langchain4j.data.message.UserMessage.userMessage;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_3_5_TURBO;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4;
import static java.time.Duration.ofSeconds;
import static java.util.stream.Collectors.joining;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.*;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.StringReader;
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
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiModelName;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
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
    public  Properties dataProp = null;
    private static final StandardAnalyzer ANALYZER = new StandardAnalyzer();

    private static final List<String> SENSITIVE_TERMS = Arrays.asList("bomb", "explosive", "terror","abusive");

    private static final Logger logger = LoggerFactory.getLogger(StepDefinitions.class);
    EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

@And("I load the data provider {string} file")
    public void getResponseFromFile(String propFileName) throws IOException {
    dataProp = new Properties();
        FileInputStream fs = new FileInputStream(
                System.getProperty("user.dir") + "//src//test//resources//Data//" +
                        propFileName + ".properties");
         dataProp.load(fs);
}
    @Then("verify the accuracy of LLM response generated with expected {string} and actual {string} response")
    public  String verify_the_accuracy_of_LLM_response(String expected, String actual) {
        if (isSensitive(dataProp.getProperty(expected)) || isSensitive(dataProp.getProperty(actual))) {
            return "I'm sorry, but I can't assist with that request.";
        }
        Map<String, Double> vectorA = textToVector(dataProp.getProperty(expected));
        Map<String, Double> vectorB = textToVector(dataProp.getProperty(actual));
        double similarity = calculateCosineSimilarity(vectorA, vectorB);
        System.out.println("Similarity: " + similarity * 100.0 + "%");
        return String.format("Similarity: %.2f%%", similarity * 100.0); // Convert similarity to percentage
    }
    // Check if the text contains sensitive terms
    private boolean isSensitive(String text) {
        return SENSITIVE_TERMS.stream().anyMatch(term -> text.toLowerCase().contains(term));
    }
    // Convert text to TF-IDF vector (map of term frequencies)
    private Map<String, Double> textToVector(String text) {
        Map<String, Double> vector = new HashMap<>();
        try (TokenStream tokenStream = ANALYZER.tokenStream("field", new StringReader(text))) {
            CharTermAttribute charTermAttribute = tokenStream.addAttribute(CharTermAttribute.class);
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
                String term = charTermAttribute.toString();
                vector.put(term, vector.getOrDefault(term, 0.0) + 1.0);
            }
            tokenStream.end();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return vector;
    }

    // Calculate cosine similarity between two term frequency vectors
    private double calculateCosineSimilarity(Map<String, Double> vectorA, Map<String, Double> vectorB) {
        Set<String> allTerms = new HashSet<>(vectorA.keySet());
        allTerms.addAll(vectorB.keySet());

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (String term : allTerms) {
            double freqA = vectorA.getOrDefault(term, 0.0);
            double freqB = vectorB.getOrDefault(term, 0.0);
            dotProduct += freqA * freqB;
            normA += freqA * freqA;
            normB += freqB * freqB;
        }

        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        if (normA == 0 || normB == 0) return 0.0;

        return dotProduct / (normA * normB);
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

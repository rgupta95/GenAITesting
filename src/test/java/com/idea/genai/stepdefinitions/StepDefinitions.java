package com.idea.genai.stepdefinitions;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static dev.langchain4j.data.message.UserMessage.userMessage;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_3_5_TURBO;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4;
import static java.time.Duration.ofSeconds;
import static java.util.stream.Collectors.joining;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.*;

import com.idea.genai.Constants;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.openai.OpenAiImageModel;
import dev.langchain4j.model.output.Response;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import java.io.*;
import java.lang.reflect.Field;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.nio.file.Paths;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
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
    public final Logger logger = LoggerFactory.getLogger(StepDefinitions.class);
    public Properties dataProp = null;
    private static final StandardAnalyzer ANALYZER = new StandardAnalyzer();
    private static final List<String> SENSITIVE_TERMS = Arrays.asList("bomb", "explosive", "terror", "abusive");
    EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

    @Given("I load the data provider {string} file")
    public void getResponseFromFile(String propFileName) throws IOException {
        dataProp = new Properties();
        FileInputStream fs = new FileInputStream(
                System.getProperty("user.dir") + "//src//test//resources//Data//" +
                        propFileName + ".properties");
        dataProp.load(fs);
    }


    @Then("verify the LLM response on sensitive question asked")
    public void verify_sensitive_input() {
        if (isSensitive(Serenity.sessionVariableCalled(Constants.SENSITIVE_PROMPT))) {
            addDataToReport("I'm sorry, but I can't assist with that request.");
        }
    }
    /*
     * Add valuable data to serenity report
     * one parameter requires- data
     * Author - Ranjan Gupta
     */
    private void addDataToReport(String data) {
        try {
            if (dataProp.containsKey(data))
                Serenity.recordReportData().withTitle("Response Printed")
                        .andContents(String.join(".", dataProp.getProperty(data)));
            else {
                Serenity.recordReportData().withTitle("Response Printed")
                        .andContents(String.join(".", data));
            }
        } catch (Exception e) {
            logger.warn(e.getMessage());
        }
    }
    @Then("verify the accuracy of characters from LLM response generated with expected {string} and actual {string} response")
    public void verify_the_accuracy_of_characters_LLM_response(String expected, String actual) throws NoSuchFieldException, IllegalAccessException {
        addDataToReport(expected);
        addDataToReport(actual);
        // Assuming 'expectedText' contains the characters you expect to find in the image
        String expectedText = "Arjun warrior sitting on his chariot with Lord Krishna";
        String extractedText = extractTextFromImage(System.getProperty("user.dir") + "//src//test//resources"+"//downloaded_image.png");
        // Convert texts to vectors (you'll need a method similar to what was described earlier)
        Map<String, Double> vectorA = textToVector(dataProp.getProperty(expectedText));
        Map<String, Double> vectorB = textToVector(extractedText);
// Calculate Cosine Similarity
        double similarity = calculateCosineSimilarity(vectorA, vectorB);
        addDataToReport(Double.toString(similarity * 100));
    }
    @Then("verify the accuracy of LLM response generated with expected {string} and actual {string} response")
    public void verify_the_accuracy_of_LLM_response(String expected, String actual) {
        addDataToReport(expected);
        addDataToReport(actual);
        Map<String, Double> vectorA = textToVector(dataProp.getProperty(expected));
        Map<String, Double> vectorB = textToVector(dataProp.getProperty(actual));
        double similarity = calculateCosineSimilarity(vectorA, vectorB);
        logger.info("Similarity: " + similarity * 100.0 + "%");
        addDataToReport(Double.toString(similarity * 100));
    }

    // Check if the text contains sensitive terms
    private boolean isSensitive(String text) {
        return SENSITIVE_TERMS.stream().anyMatch(term -> text.toLowerCase().contains(term));
    }
    /*
     * Creates vector of text using Apache lucene library
     * one parameter requires- text
     * Author - Ranjan Gupta
     */
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
    /*
     * calculates cosine similarity between 2 vectors
     * two parameter requires- vectpr A, vector B
     * Author - Ranjan Gupta
     */
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


    /*
    * Extraction of text from image using Tesseract library
    * one parameter requires- image path
    * not ready yet
    * Author - Ranjan Gupta
     */
    public String extractTextFromImage(String imagePath) throws NoSuchFieldException, IllegalAccessException {
        System.setProperty("java.library.path", "/usr/local/Cellar/tesseract/5.4.1/lib");
        // Use reflection to reset the library path
        Field fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
        fieldSysPath.setAccessible(true);
        fieldSysPath.set(null, null);
        System.loadLibrary("tesseract");
        Tesseract tesseract = new Tesseract();
        tesseract.setDatapath(System.getProperty("user.dir") + "/src/test/resources"+"/downloaded_image.png");  // Set the path to tessdata folder
        try {
            return tesseract.doOCR(new File(imagePath));
        } catch (TesseractException e) {
            e.printStackTrace();
            return null;
        }
    }
    @Given("Generate image from Open AI llm model with prompt {string}")
    public void generate_image_From_OpenAi_model(String prompt) {
        ImageModel model = OpenAiImageModel.withApiKey(Constants.OPENAI_API_KEY);
        Response<Image> imageResponse = model.generate(prompt);
        Serenity.setSessionVariable(Constants.IMAGE_PROMPT_RESPONSE).to(imageResponse);
        addDataToReport(imageResponse.content().toString());
        String imageUrl = imageResponse.content().url().toString();  // Assuming getUrl() provides the image URL
        Path targetPath = Paths.get(System.getProperty("user.dir") + "//src//test//resources"+"//downloaded_image.png");
        try (InputStream in = new URL(imageUrl).openStream()) {
            Files.copy(in, targetPath, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            e.printStackTrace();
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
        addDataToReport(response);
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
        addDataToReport(data);
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
        logger.info("Chat memory is created" + chatMemory);
        addDataToReport(chatMemory.toString());
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
        Serenity.setSessionVariable(Constants.SENSITIVE_PROMPT).to(data);
        addDataToReport(data);
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

Feature: POC on Gen AI testings - Feature includes multiple use cases to assert LLM responses



  Scenario Outline: Assert  LLM responses based on prompt variable
    Given Generate prompt template with variable <adjective> about <content>

    Then verify the LLM response generated for response "prompt_response"

    Examples:
      | adjective      | content |
      | funny          | MSDhoni |
      | jobOpportunity | SDET    |
      | brief          | Y2K     |


  Scenario: Create a conversation with LLM  and assert the responses

    Given I created chat memory with LLM model
    And I add input data "Hello, my name is Ranjan Gupta" to chat memory
    Then verify the LLM response generated from chat memory "chat_memory_response"
    And I add input data "What is my name" to chat memory
    Then verify the LLM response contains "Gupta"


  Scenario: Example of RAG -Retrieval Augmented Generation use case with assertions
    Given I upload a document "inputData.txt" and create word embeddings
    Then I ask LLM question "What was the year when Y2K bug found" based on stored word embeddings
  Then verify the LLM response contains "1990"
    Given I created chat memory with LLM model
    And I add input data "Brief me about Y2K in 100 words" to chat memory
    ######### Assertion catches defect in response ######################
  Then verify the LLM response contains "100" words

#@test
#  Scenario: Create a conversation with LLM  and assert the accuracy of response based on response criteria
#    Given I created chat memory with LLM model
#    And I add input data "Explain the benefits of AI in automation testing" to chat memory
#    Then verify the accuracy of LLM response generated with response criteria having values like matching keywords "AI automation testing benefits accuracy efficiency coverage maintenance costs faster" , minimum length "50", punctuation ". ? !" and expected response "AI in automation testing provides several benefits including increased accuracy, enhanced efficiency, broader test coverage, reduced maintenance efforts, and lower testing costs. These advantages help deliver faster and more reliable testing outcomes."
#    Then verify the accuracy of LLM response generated with response criteria having values like matching keywords "Continuous testing Cost-effective Predictive analytics accuracy efficiency coverage maintenance costs faster" , minimum length "50", punctuation ". ? !" and expected response "AI in automation testing provides several benefits including increased accuracy, enhanced efficiency, Scalability ,reduced maintenance efforts, and lower testing costs.These advantages help deliver faster and more reliable testing outcomes."
#    Then verify the accuracy of LLM response generated with response criteria having values like matching keywords "Continuous testing Cost-effective Predictive analytics Improved accuracy Increased efficiency" , minimum length "50", punctuation ". ? !" and expected response "AI in automation testing provides several benefits including increased accuracy, enhanced efficiency, Scalability ,reduced maintenance efforts, Cost-effective,and lower testing costs."
  @test
  Scenario: Create a conversation with LLM  and assert the accuracy of response based on response criteria
    Given I created chat memory with LLM model
    And I add input data "Explain the benefits of AI in automation testing" to chat memory
    Then verify the accuracy of LLM response generated with response criteria having values like matching keywords "AI automation testing" , minimum length "50", punctuation ". ? !" and expected response "AI in automation testing provides several benefits including increased accuracy, enhanced efficiency, broader test coverage, reduced maintenance efforts, and lower testing costs. These advantages help deliver faster and more reliable testing outcomes."
    Then verify the accuracy of LLM response generated with response criteria having values like matching keywords "Continuous testing Cost-effective Predictive analytics increased accuracy efficiency coverage maintenance costs faster" , minimum length "50", punctuation ". ? !" and expected response "AI in automation testing provides several benefits including increased accuracy, enhanced efficiency, broader test coverage, reduced maintenance efforts, and lower testing costs. These advantages help deliver faster and more reliable testing outcomes."
    Then verify the accuracy of LLM response generated with response criteria having values like matching keywords "Continuous testing Cost-effective reduced maintenance efforts Predictive analytics Improved accuracy Increased efficiency" , minimum length "50", punctuation ". ? !" and expected response "AI in automation testing provides several benefits including increased accuracy, enhanced efficiency, broader test coverage, reduced maintenance efforts, and lower testing costs. These advantages help deliver faster and more reliable testing outcomes."
